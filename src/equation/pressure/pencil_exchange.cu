#include "equation/pressure/pencil_transpose.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"

// GPU 백엔드 — PaScaL_TDMA(pascal_tdma_cuda.cu, alltoallv_double)와 같은 구조:
//   팩 커널(디바이스) → 디바이스 송신 버퍼를 **그대로 MPI_Alltoallv** (CUDA-aware MPI, GPU↔GPU)
//   → 디바이스 수신 버퍼 → 언팩 커널. host_staging(PASCAL_TDMA_MPI_MODE=host)일 때만 호스트 경유.

namespace mpmstd::equation::pressure {

namespace {

enum : int { kThreads = 256 };

__global__ void gather_kernel(real_t* send, const real_t* src, const int* index, const long long n,
                              const int width, const real_t scale, const real_t subtract) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= n * width) return;
  const long long e = q / width;
  const int c = static_cast<int>(q - e * width);
  send[q] = src[static_cast<long long>(index[e]) * width + c] * scale - subtract;
}

__global__ void scatter_kernel(real_t* dst, const real_t* recv, const int* index, const int* mirror,
                               const long long n, const int width) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= n * width) return;
  const long long e = q / width;
  const int c = static_cast<int>(q - e * width);
  dst[static_cast<long long>(index[e]) * width + c] = recv[q];
  if (mirror) dst[static_cast<long long>(mirror[e]) * width + c] = recv[q];
}

int* upload_indices(const std::vector<int>& host) {
  if (host.empty()) return nullptr;
  int* device = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(int)));
  MPMSTD_CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
  return device;
}

int blocks_for(const long long n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

// [2026-11-02] self 블록을 cudaMemcpyAsync 로 (PaScaL alltoallv_double 과 같은 스위치) — cray-mpich 9.0.1(GTL)은
// 노드당 랭크가 여럿이면 디바이스 버퍼 self 전송이 ~1 GB/s (Debug/bug.md). MPMSTD_A2A_SELF=mpi 면 원래 경로.
bool self_copy_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MPMSTD_A2A_SELF");
    return !(value != nullptr && std::strcmp(value, "mpi") == 0);
  }();
  return enabled;
}

// ── [최적화 2026-11-02] 구조화 팩·언팩 (MPMSTD_PENCIL_PACK=index 면 위의 인덱스 표 경로) ──────────────
// 인덱스 표 gather/scatter 는 전치하는 쪽이 원소마다 보폭 접근 + int 표 읽기라 slab 의 같은 전치보다 ~10배 느렸다
// (np 8 펜슬, 전치 4 종의 팩·언팩 합 ~12 ms vs slab 대응 커널 ~1.4 ms). 상대 r 의 y 조각이 y 순서로 이어지므로
// 전치하는 쪽의 송·수신 버퍼 전체가 연속 배열 하나가 된다 → slab 과 같은 32×32 타일 전치 한 번.
// 반대쪽(r 마다 x 구간 / 모드 구간)은 행 단위 연속 복사. 같은 값을 같은 연산으로 옮기므로 비트 동일.
constexpr int kTile = 32;
constexpr int kTileRows = 8;

// 장 (x,y,kz0+k) k 연속 → 버퍼 [(yl·nz + k)·bxn + xl] x 연속 (× scale − subtract), k < nz. 그리드 (⌈bxn/32⌉, ⌈nz/32⌉, byn)
__global__ void field_to_rows_kernel(real_t* buf, const real_t* f, const int h, const int nyt, const int nzt,
                                     const int kz0, const int bxn, const int nz, const real_t scale,
                                     const real_t subtract) {
  __shared__ real_t tile[kTile][kTile + 1];
  const int yl = static_cast<int>(blockIdx.z);
  const int x0 = static_cast<int>(blockIdx.x) * kTile, k0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {          // 읽기: tx 가 k 를 걷는다
    const int xl = x0 + ty + q, k = k0 + tx;
    if (xl < bxn && k < nz)
      tile[ty + q][tx] = f[(static_cast<std::size_t>(xl + h) * nyt + (yl + h)) * nzt + (kz0 + k + h)];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {          // 쓰기: tx 가 x 를 걷는다
    const int xl = x0 + tx, k = k0 + ty + q;
    if (xl < bxn && k < nz)
      buf[(static_cast<std::size_t>(yl) * nz + k) * bxn + xl] = tile[tx][ty + q] * scale - subtract;
  }
}

// 역: 버퍼 [(yl·nz + k)·bxn + xl] → 장 (x,y,kz0+k). 그리드 (⌈nz/32⌉, ⌈bxn/32⌉, byn)
__global__ void rows_to_field_kernel(real_t* f, const real_t* buf, const int h, const int nyt, const int nzt,
                                     const int kz0, const int bxn, const int nz) {
  __shared__ real_t tile[kTile][kTile + 1];
  const int yl = static_cast<int>(blockIdx.z);
  const int k0 = static_cast<int>(blockIdx.x) * kTile, x0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {          // 읽기: tx 가 x 를 걷는다
    const int k = k0 + ty + q, xl = x0 + tx;
    if (xl < bxn && k < nz) tile[ty + q][tx] = buf[(static_cast<std::size_t>(yl) * nz + k) * bxn + xl];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {          // 쓰기: tx 가 k 를 걷는다
    const int k = k0 + tx, xl = x0 + ty + q;
    if (xl < bxn && k < nz)
      f[(static_cast<std::size_t>(xl + h) * nyt + (yl + h)) * nzt + (kz0 + k + h)] = tile[tx][ty + q];
  }
}

// 복소 [(y·nz + k)·nml + m] (m 연속) → [(k·nml + m)·ny + y] (y 연속). 그리드 (⌈nml/32⌉, ⌈ny/32⌉, nz)
__global__ void ykm_to_kmy_kernel(real_t* dst_r, const real_t* src_r, const int ny, const int nz, const int nml) {
  __shared__ double2 tile[kTile][kTile + 1];
  const double2* src = reinterpret_cast<const double2*>(src_r);
  double2* dst = reinterpret_cast<double2*>(dst_r);
  const int k = static_cast<int>(blockIdx.z);
  const int m0 = static_cast<int>(blockIdx.x) * kTile, y0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {
    const int m = m0 + tx, y = y0 + ty + q;
    if (m < nml && y < ny) tile[ty + q][tx] = src[(static_cast<std::size_t>(y) * nz + k) * nml + m];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {
    const int m = m0 + ty + q, y = y0 + tx;
    if (m < nml && y < ny) dst[(static_cast<std::size_t>(k) * nml + m) * ny + y] = tile[tx][ty + q];
  }
}

// 역: [(k·nml + m)·ny + y] → [(y·nz + k)·nml + m]. 그리드 (⌈ny/32⌉, ⌈nml/32⌉, nz)
__global__ void kmy_to_ykm_kernel(real_t* dst_r, const real_t* src_r, const int ny, const int nz, const int nml) {
  __shared__ double2 tile[kTile][kTile + 1];
  const double2* src = reinterpret_cast<const double2*>(src_r);
  double2* dst = reinterpret_cast<double2*>(dst_r);
  const int k = static_cast<int>(blockIdx.z);
  const int y0 = static_cast<int>(blockIdx.x) * kTile, m0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {
    const int y = y0 + tx, m = m0 + ty + q;
    if (m < nml && y < ny) tile[ty + q][tx] = src[(static_cast<std::size_t>(k) * nml + m) * ny + y];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {
    const int y = y0 + ty + q, m = m0 + tx;
    if (m < nml && y < ny) dst[(static_cast<std::size_t>(y) * nz + k) * nml + m] = tile[tx][ty + q];
  }
}

// 행 복사: dst[row·dst_stride + c] = src[row·src_stride + c] × scale − subtract (c < len, double 단위).
// mirror_line > 0 이면 거울 라인 dst[row·dst_stride + (mirror_line − 1 − (mirror_col0 + c)) − mirror_col0] 에도
// (dst 는 라인 시작 + mirror_col0 을 가리킨다 — 실수 라인 전용).
__global__ void copy_rows_kernel(real_t* dst, const long long dst_stride, const real_t* src,
                                 const long long src_stride, const long long n_rows, const int len,
                                 const real_t scale, const real_t subtract, const int mirror_line,
                                 const int mirror_col0) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= n_rows * len) return;
  const long long row = q / len;
  const int c = static_cast<int>(q - row * len);
  const real_t v = src[row * src_stride + c] * scale - subtract;
  dst[row * dst_stride + c] = v;
  if (mirror_line > 0) dst[row * dst_stride + (mirror_line - 1 - (mirror_col0 + c)) - mirror_col0] = v;
}

void copy_rows(real_t* dst, const long long dst_stride, const real_t* src, const long long src_stride,
               const long long n_rows, const int len, const real_t scale = 1, const real_t subtract = 0,
               const int mirror_line = 0, const int mirror_col0 = 0) {
  if (n_rows <= 0 || len <= 0) return;
  copy_rows_kernel<<<blocks_for(n_rows * len), kThreads>>>(dst, dst_stride, src, src_stride, n_rows, len,
                                                           scale, subtract, mirror_line, mirror_col0);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

int overlap(const int a0, const int an, const int b0, const int bn, int& start) {
  start = a0 > b0 ? a0 : b0;
  const int end = (a0 + an < b0 + bn) ? a0 + an : b0 + bn;
  return end > start ? end - start : 0;
}

bool structured_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MPMSTD_PENCIL_PACK");
    return !(value != nullptr && std::strcmp(value, "index") == 0);
  }();
  return enabled;
}

bool use_structured(const PencilExchange& e) {
  return e.kind != PencilKind::generic && (e.layout != nullptr || e.planes != nullptr) && structured_enabled();
}

// 평면 묶음 전치 (PlaneLayout) — 송신 (블록 → 평면): g 마다 내 블록 (y, g 의 평면, x) 를 타일 전치.
// 수신: r 마다 r 블록 (y, 내 평면, x) 가 평면 라인의 연속 행 [by0_r·nzp, (by0_r+byn_r)·nzp) 의 x 구간 → 행 복사.
void pack_planes(const PencilExchange& e, const real_t* src, real_t* send, const real_t scale,
                 const real_t subtract) {
  const PlaneLayout& L = *e.planes;
  const int me = L.rank, nzp = L.my_plane_count();
  const int bxn = L.block_x_count[me], byn = L.block_y_count[me];
  const dim3 threads(kTile, kTileRows);
  if (e.kind == PencilKind::block_to_planes) {
    for (int g = 0; g < L.size; ++g) {
      const int nk = L.plane_count[g];
      if (nk == 0) continue;
      const dim3 blocks((bxn + kTile - 1) / kTile, (nk + kTile - 1) / kTile, byn);
      field_to_rows_kernel<<<blocks, threads>>>(send + e.send_displs[g], src, L.halo, L.ny_total, L.nz_total,
                                                L.plane_offset[g], bxn, nk, scale, subtract);
      MPMSTD_CUDA_CHECK_LAUNCH();
    }
  } else {   // planes_to_block: r 마다 평면 라인의 r 블록 구간 → (y, kk, x)
    for (int r = 0; r < L.size; ++r) {
      if (L.block_x_count[r] == 0 || L.block_y_count[r] == 0) continue;
      copy_rows(send + e.send_displs[r], L.block_x_count[r],
                src + static_cast<std::size_t>(L.block_y_offset[r]) * nzp * L.line + L.block_x_offset[r], L.line,
                static_cast<long long>(L.block_y_count[r]) * nzp, L.block_x_count[r], scale, subtract);
    }
  }
}

void unpack_planes(const PencilExchange& e, const real_t* recv, real_t* dst) {
  const PlaneLayout& L = *e.planes;
  const int me = L.rank, nzp = L.my_plane_count();
  const dim3 threads(kTile, kTileRows);
  if (e.kind == PencilKind::block_to_planes) {
    for (int r = 0; r < L.size; ++r) {
      if (L.block_x_count[r] == 0 || L.block_y_count[r] == 0) continue;
      const int x0 = L.block_x_offset[r];
      copy_rows(dst + static_cast<std::size_t>(L.block_y_offset[r]) * nzp * L.line + x0, L.line,
                recv + e.recv_displs[r], L.block_x_count[r], static_cast<long long>(L.block_y_count[r]) * nzp,
                L.block_x_count[r], 1, 0, L.mirrored ? L.line : 0, x0);
    }
  } else {   // planes_to_block: g 마다 (y, g 의 평면, x) → 장
    const int bxn = L.block_x_count[me], byn = L.block_y_count[me];
    for (int g = 0; g < L.size; ++g) {
      const int nk = L.plane_count[g];
      if (nk == 0) continue;
      const dim3 blocks((nk + kTile - 1) / kTile, (bxn + kTile - 1) / kTile, byn);
      rows_to_field_kernel<<<blocks, threads>>>(dst, recv + e.recv_displs[g], L.halo, L.ny_total, L.nz_total,
                                                L.plane_offset[g], bxn, nk);
      MPMSTD_CUDA_CHECK_LAUNCH();
    }
  }
}

// 송신 버퍼 채우기 (pencil_layout.cpp 의 send_index 와 같은 순서·값)
void pack_structured(const PencilExchange& e, const real_t* src, real_t* send, const real_t scale,
                     const real_t subtract) {
  if (e.kind == PencilKind::block_to_planes || e.kind == PencilKind::planes_to_block) {
    pack_planes(e, src, send, scale, subtract);
    return;
  }
  const PencilLayout& L = *e.layout;
  const int nz = L.nz_local, me = L.rank;
  const dim3 threads(kTile, kTileRows);
  switch (e.kind) {
    case PencilKind::block_to_lines: {   // 장 → (y, k, x) 블록 전체 = r 조각의 연결
      const int bxn = L.block_x_count[me], byn = L.block_y_count[me];
      const dim3 blocks((bxn + kTile - 1) / kTile, (nz + kTile - 1) / kTile, byn);
      field_to_rows_kernel<<<blocks, threads>>>(send, src, L.halo, L.ny_total, L.nz_total, 0, bxn, nz, scale,
                                                subtract);
      MPMSTD_CUDA_CHECK_LAUNCH();
      break;
    }
    case PencilKind::lines_to_block: {   // r 마다: 내 y 조각 ∩ r 블록 y × k × r 블록 x 구간
      for (int r = 0; r < L.size; ++r) {
        int y0 = 0;
        const int ny = overlap(L.block_y_offset[r], L.block_y_count[r], L.my_y_offset(), L.my_y_count(), y0);
        if (ny == 0 || L.block_x_count[r] == 0) continue;
        copy_rows(send + e.send_displs[r], L.block_x_count[r],
                  src + static_cast<std::size_t>(y0 - L.my_y_offset()) * nz * L.line + L.block_x_offset[r],
                  L.line, static_cast<long long>(ny) * nz, L.block_x_count[r], scale, subtract);
      }
      break;
    }
    case PencilKind::x_to_y: {           // r 마다: 내 y 조각 × k × r 모드 구간 (복소)
      const long long rows = static_cast<long long>(L.my_y_count()) * nz;
      for (int r = 0; r < L.size; ++r) {
        if (L.mode_count[r] == 0) continue;
        copy_rows(send + e.send_displs[r], 2LL * L.mode_count[r], src + 2LL * L.mode_offset[r],
                  2LL * L.n_mode_x, rows, 2 * L.mode_count[r], scale, subtract);
      }
      break;
    }
    case PencilKind::y_to_x: {           // (k, m, y) → (y 전역, k, m) = r 조각의 연결
      const int nml = L.my_mode_count();
      const dim3 blocks((L.n_y + kTile - 1) / kTile, (nml + kTile - 1) / kTile, nz);
      kmy_to_ykm_kernel<<<blocks, threads>>>(send, src, L.n_y, nz, nml);
      MPMSTD_CUDA_CHECK_LAUNCH();
      break;
    }
    case PencilKind::block_to_planes:   // 위에서 먼저 처리
    case PencilKind::planes_to_block:
      break;
    case PencilKind::generic:
      break;
  }
}

// 수신 버퍼 풀기 (recv_index / recv_mirror 와 같은 순서·값)
void unpack_structured(const PencilExchange& e, const real_t* recv, real_t* dst) {
  if (e.kind == PencilKind::block_to_planes || e.kind == PencilKind::planes_to_block) {
    unpack_planes(e, recv, dst);
    return;
  }
  const PencilLayout& L = *e.layout;
  const int nz = L.nz_local, me = L.rank;
  const dim3 threads(kTile, kTileRows);
  switch (e.kind) {
    case PencilKind::block_to_lines: {   // r 마다: r 블록 y ∩ 내 y 조각 × k × r 블록 x → x 라인 (거울이면 반대편도)
      for (int r = 0; r < L.size; ++r) {
        int s0 = 0;
        const int ry = overlap(L.block_y_offset[r], L.block_y_count[r], L.my_y_offset(), L.my_y_count(), s0);
        if (ry == 0 || L.block_x_count[r] == 0) continue;
        const int x0 = L.block_x_offset[r];
        copy_rows(dst + static_cast<std::size_t>(s0 - L.my_y_offset()) * nz * L.line + x0, L.line,
                  recv + e.recv_displs[r], L.block_x_count[r], static_cast<long long>(ry) * nz,
                  L.block_x_count[r], 1, 0, L.mirrored ? L.line : 0, x0);
      }
      break;
    }
    case PencilKind::lines_to_block: {   // (y, k, x) 블록 전체 → 장
      const int bxn = L.block_x_count[me], byn = L.block_y_count[me];
      const dim3 blocks((nz + kTile - 1) / kTile, (bxn + kTile - 1) / kTile, byn);
      rows_to_field_kernel<<<blocks, threads>>>(dst, recv, L.halo, L.ny_total, L.nz_total, 0, bxn, nz);
      MPMSTD_CUDA_CHECK_LAUNCH();
      break;
    }
    case PencilKind::x_to_y: {           // (y 전역, k, m) → (k, m, y)
      const int nml = L.my_mode_count();
      const dim3 blocks((nml + kTile - 1) / kTile, (L.n_y + kTile - 1) / kTile, nz);
      ykm_to_kmy_kernel<<<blocks, threads>>>(dst, recv, L.n_y, nz, nml);
      MPMSTD_CUDA_CHECK_LAUNCH();
      break;
    }
    case PencilKind::y_to_x: {           // r 마다: 내 y 조각 × k × r 모드 구간 → x 스펙트럼
      const long long rows = static_cast<long long>(L.my_y_count()) * nz;
      for (int r = 0; r < L.size; ++r) {
        if (L.mode_count[r] == 0) continue;
        copy_rows(dst + 2LL * L.mode_offset[r], 2LL * L.n_mode_x, recv + e.recv_displs[r],
                  2LL * L.mode_count[r], rows, 2 * L.mode_count[r]);
      }
      break;
    }
    case PencilKind::block_to_planes:   // 위에서 먼저 처리
    case PencilKind::planes_to_block:
      break;
    case PencilKind::generic:
      break;
  }
}

}  // namespace

void prepare_exchange(PencilExchange& e) {
  if (!use_structured(e)) {   // 구조화 경로는 인덱스 표를 디바이스에 올리지 않는다 (표 3 개 ≈ 장 1.5 개 분량)
    e.device_send_index = upload_indices(e.send_index);
    e.device_recv_index = upload_indices(e.recv_index);
    e.device_recv_mirror = upload_indices(e.recv_mirror);
  }
  // 크기 0 버퍼도 유효해야 하므로 최소 1 원소
  e.send_buffer.assign(std::max<std::size_t>(1, e.send_index.size() * e.width));
  e.recv_buffer.assign(std::max<std::size_t>(1, e.recv_index.size() * e.width));
}

void release_exchange(PencilExchange& e) {
  for (int** p : {&e.device_send_index, &e.device_recv_index, &e.device_recv_mirror}) {
    if (*p) cudaFree(*p);
    *p = nullptr;
  }
}

void run_exchange(PencilExchange& e, const real_t* src, real_t* dst, MPI_Comm comm,
                  const bool host_staging, const real_t scale, const real_t subtract) {
  const long long n_send = static_cast<long long>(e.send_index.size());
  const long long n_recv = static_cast<long long>(e.recv_index.size());
  real_t* send = e.send_buffer.data();
  real_t* recv = e.recv_buffer.data();
  const bool structured = use_structured(e);
  // y_to_x 팩(타일 전치)은 scale·subtract 를 적용하지 않는다 — solve.cpp 는 기본값만 넘긴다 (조용한 오답 금지)
  if (structured && e.kind == PencilKind::y_to_x && (scale != 1 || subtract != 0))
    throw std::invalid_argument("pressure pencil: y_to_x structured pack does not apply scale/subtract");
  if (structured) {
    pack_structured(e, src, send, scale, subtract);
  } else if (n_send > 0) {
    gather_kernel<<<blocks_for(n_send * e.width), kThreads>>>(send, src, e.device_send_index, n_send,
                                                              e.width, scale, subtract);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  // 디바이스 버퍼를 MPI 에 넘기기 직전 — 팩 커널 완료를 보장 (PaScaL 의 cudaStreamSynchronize 와 같은 자리)
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
  const double mpi_t0 = MPI_Wtime();
  int status = MPI_SUCCESS;
  if (!host_staging && self_copy_enabled()) {
    int me = 0;
    MPI_Comm_rank(comm, &me);
    if (e.send_counts[me] != e.recv_counts[me])
      throw std::runtime_error("pressure pencil: self send/recv counts differ");
    if (e.send_counts[me] > 0)
      MPMSTD_CUDA_CHECK(cudaMemcpyAsync(recv + e.recv_displs[me], send + e.send_displs[me],
                                        static_cast<std::size_t>(e.send_counts[me]) * sizeof(real_t),
                                        cudaMemcpyDeviceToDevice));   // 기본 스트림 — 아래 scatter 커널보다 먼저
    std::vector<int> peer_send_counts(e.send_counts), peer_recv_counts(e.recv_counts);
    peer_send_counts[me] = 0;
    peer_recv_counts[me] = 0;
    status = MPI_Alltoallv(send, peer_send_counts.data(), e.send_displs.data(), MPI_DOUBLE, recv,
                           peer_recv_counts.data(), e.recv_displs.data(), MPI_DOUBLE, comm);
  } else if (!host_staging) {
    status = MPI_Alltoallv(send, e.send_counts.data(), e.send_displs.data(), MPI_DOUBLE, recv,
                           e.recv_counts.data(), e.recv_displs.data(), MPI_DOUBLE, comm);
  } else {
    std::vector<real_t> send_host(e.send_buffer.size()), recv_host(e.recv_buffer.size());
    MPMSTD_CUDA_CHECK(cudaMemcpy(send_host.data(), send, send_host.size() * sizeof(real_t),
                                 cudaMemcpyDeviceToHost));
    status = MPI_Alltoallv(send_host.data(), e.send_counts.data(), e.send_displs.data(), MPI_DOUBLE,
                           recv_host.data(), e.recv_counts.data(), e.recv_displs.data(), MPI_DOUBLE,
                           comm);
    MPMSTD_CUDA_CHECK(cudaMemcpy(recv, recv_host.data(), recv_host.size() * sizeof(real_t),
                                 cudaMemcpyHostToDevice));
  }
  e.comm_seconds += MPI_Wtime() - mpi_t0;   // self 복사(비동기)는 들어가지 않는다
  if (status != MPI_SUCCESS) throw std::runtime_error("pressure pencil: MPI_Alltoallv failed");
  if (structured) {
    unpack_structured(e, recv, dst);
  } else if (n_recv > 0) {
    scatter_kernel<<<blocks_for(n_recv * e.width), kThreads>>>(dst, recv, e.device_recv_index,
                                                               e.device_recv_mirror, n_recv, e.width);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace mpmstd::equation::pressure
