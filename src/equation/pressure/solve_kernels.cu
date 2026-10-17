#include "equation/pressure/solve_kernels.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// GPU 백엔드 — solve_kernels.cpp 의 쌍둥이.

namespace mpmstd::equation::pressure {

namespace {

enum : int { kThreads = 256 };

inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

// [최적화 2026-10-18] 배치 전환 4개(필드 ↔ x 라인, x 스펙트럼 ↔ y 배치)는 전치 복사 — 1차원 매핑은 한쪽이 보폭(이상값의 3~7배).
//   32×32 공유메모리 타일로 양쪽 연속 접근. 복사뿐이라 비트 동일.
constexpr int kTile = 32;
constexpr int kTileRows = 8;

// 필드 (i,j,k) k 연속 → 라인 [(jj·nz + kk)·n_x + ii] i 연속: jj 마다 (ii,kk) 전치. 그리드 (⌈n_x/32⌉, ⌈nz/32⌉, n_y)
__global__ void pack_rhs_kernel(real_t* lines, const field::ConstView r, const int h,
                                const int n_x, const int n_y, const int nz_local) {
  __shared__ real_t tile[kTile][kTile + 1];
  const int jj = static_cast<int>(blockIdx.z);
  const int i0 = static_cast<int>(blockIdx.x) * kTile, k0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {          // 읽기: tx 가 k 를 걷는다
    const int ii = i0 + ty + q, kk = k0 + tx;
    if (ii < n_x && kk < nz_local) tile[ty + q][tx] = r(ii + h, jj + h, kk + h);
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {          // 쓰기: tx 가 i 를 걷는다
    const int ii = i0 + tx, kk = k0 + ty + q;
    if (ii < n_x && kk < nz_local)
      lines[(static_cast<std::size_t>(jj) * nz_local + kk) * n_x + ii] = tile[tx][ty + q];
  }
}

// x 스펙트럼 [(jy·nz + kk)·Nxm + mx] (mx 연속) → y 배치 [(kk·Nxm + mx)·Ny + jy] (jy 연속): kk 마다 (jy,mx) 전치.
// 복소 (실,허) 쌍은 double2 로 한 번에. 그리드 (⌈Nxm/32⌉, ⌈Ny/32⌉, nz)
__global__ void reorder_x_to_y_kernel(real_t* sxy, const real_t* sx, const int n_y,
                                      const int nz_local, const int n_mode_x) {
  __shared__ double2 tile[kTile][kTile + 1];
  const double2* src = reinterpret_cast<const double2*>(sx);
  double2* dst = reinterpret_cast<double2*>(sxy);
  const int kk = static_cast<int>(blockIdx.z);
  const int m0 = static_cast<int>(blockIdx.x) * kTile, j0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {          // 읽기: tx 가 mx 를 걷는다
    const int mx = m0 + tx, jy = j0 + ty + q;
    if (mx < n_mode_x && jy < n_y)
      tile[ty + q][tx] = src[(static_cast<std::size_t>(jy) * nz_local + kk) * n_mode_x + mx];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {          // 쓰기: tx 가 jy 를 걷는다
    const int mx = m0 + ty + q, jy = j0 + tx;
    if (mx < n_mode_x && jy < n_y)
      dst[(static_cast<std::size_t>(kk) * n_mode_x + mx) * n_y + jy] = tile[tx][ty + q];
  }
}

// 역방향: y 배치 (jy 연속) → x 스펙트럼 (mx 연속). 그리드 (⌈Ny/32⌉, ⌈Nxm/32⌉, nz)
__global__ void reorder_y_to_x_kernel(real_t* sx, const real_t* sxy, const int n_y,
                                      const int nz_local, const int n_mode_x) {
  __shared__ double2 tile[kTile][kTile + 1];
  const double2* src = reinterpret_cast<const double2*>(sxy);
  double2* dst = reinterpret_cast<double2*>(sx);
  const int kk = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.x) * kTile, m0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {          // 읽기: tx 가 jy 를 걷는다
    const int jy = j0 + tx, mx = m0 + ty + q;
    if (mx < n_mode_x && jy < n_y)
      tile[ty + q][tx] = src[(static_cast<std::size_t>(kk) * n_mode_x + mx) * n_y + jy];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {          // 쓰기: tx 가 mx 를 걷는다
    const int jy = j0 + ty + q, mx = m0 + tx;
    if (mx < n_mode_x && jy < n_y)
      dst[(static_cast<std::size_t>(jy) * nz_local + kk) * n_mode_x + mx] = tile[tx][ty + q];
  }
}

__global__ void pack_bands_kernel(real_t* D, const real_t* sxy, const int nz_local,
                                  const int n_mode_x, const int n_y,
                                  const bool pin_mean_row) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(nz_local) * n_mode_x * n_y) return;
  const int jy = static_cast<int>(idx % n_y);
  const int mx = static_cast<int>((idx / n_y) % n_mode_x);
  const int kk = static_cast<int>(idx / (static_cast<long long>(n_y) * n_mode_x));
  const int nsys = 2 * n_mode_x * n_y;
  const std::size_t q = 2 * ((static_cast<std::size_t>(kk) * n_mode_x + mx) * n_y + jy);
  const std::size_t p = static_cast<std::size_t>(kk) * nsys
                        + 2 * (static_cast<std::size_t>(mx) * n_y + jy);
  const bool pinned = pin_mean_row && kk == 0 && mx == 0 && jy == 0;
  D[p] = pinned ? 0 : sxy[q];
  D[p + 1] = pinned ? 0 : sxy[q + 1];
}

__global__ void unpack_bands_kernel(real_t* sxy, const real_t* D, const int nz_local,
                                    const int n_mode_x, const int n_y) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(nz_local) * n_mode_x * n_y) return;
  const int jy = static_cast<int>(idx % n_y);
  const int mx = static_cast<int>((idx / n_y) % n_mode_x);
  const int kk = static_cast<int>(idx / (static_cast<long long>(n_y) * n_mode_x));
  const int nsys = 2 * n_mode_x * n_y;
  const std::size_t q = 2 * ((static_cast<std::size_t>(kk) * n_mode_x + mx) * n_y + jy);
  const std::size_t p = static_cast<std::size_t>(kk) * nsys
                        + 2 * (static_cast<std::size_t>(mx) * n_y + jy);
  sxy[q] = D[p];
  sxy[q + 1] = D[p + 1];
}

// 라인 (i 연속) → 필드 (k 연속), 정규화 배율 포함. 그리드 (⌈nz/32⌉, ⌈n_x/32⌉, n_y)
__global__ void scatter_lines_kernel(const field::View p, const real_t* lines,
                                     const real_t scale, const int h, const int n_x,
                                     const int n_y, const int nz_local) {
  __shared__ real_t tile[kTile][kTile + 1];
  const int jj = static_cast<int>(blockIdx.z);
  const int k0 = static_cast<int>(blockIdx.x) * kTile, i0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  for (int q = 0; q < kTile; q += kTileRows) {          // 읽기: tx 가 i 를 걷는다
    const int ii = i0 + tx, kk = k0 + ty + q;
    if (ii < n_x && kk < nz_local)
      tile[ty + q][tx] = lines[(static_cast<std::size_t>(jj) * nz_local + kk) * n_x + ii];
  }
  __syncthreads();
  for (int q = 0; q < kTile; q += kTileRows) {          // 쓰기: tx 가 k 를 걷는다
    const int ii = i0 + ty + q, kk = k0 + tx;
    if (ii < n_x && kk < nz_local) p(ii + h, jj + h, kk + h) = tile[tx][ty + q] * scale;
  }
}

__global__ void interior_sum_kernel(const field::ConstView p, const int h,
                                    const int i1, const int i2, const int i3,
                                    real_t* total) {
  __shared__ real_t partial[kThreads];
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  real_t mine = 0;
  if (idx < static_cast<long long>(i1) * i2 * i3) {
    const int k = h + static_cast<int>(idx % i3);
    const int j = h + static_cast<int>((idx / i3) % i2);
    const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
    mine = p(i, j, k);
  }
  partial[threadIdx.x] = mine;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

__global__ void subtract_kernel(const field::View p, const int h, const int i1,
                                const int i2, const int i3, const real_t value) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  p(i, j, k) -= value;
}




// ── x Neumann(DCT) 경로 커널 (solve_kernels.hpp 의 식; CPU 쌍둥이와 같은 산술) ──
__global__ void pack_mirrored_kernel(real_t* lines, const field::ConstView r, const int h,
                                     const int n_x, const int n_y, const int nz_local) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_x) * n_y * nz_local) return;
  const int ii = static_cast<int>(idx % n_x);
  const int kk = static_cast<int>((idx / n_x) % nz_local);
  const int jj = static_cast<int>(idx / (static_cast<long long>(n_x) * nz_local));
  const int n2 = 2 * n_x;
  real_t* line = lines + (static_cast<std::size_t>(jj) * nz_local + kk) * n2;
  const real_t v = r(ii + h, jj + h, kk + h);
  line[ii] = v;
  line[n2 - 1 - ii] = v;  // 거울
}

__global__ void dct_forward_kernel(real_t* sx, const real_t* ms, const int n_lines,
                                   const int n_x, const real_t pi) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_lines) * n_x) return;
  const int k = static_cast<int>(idx % n_x);
  const int line = static_cast<int>(idx / n_x);
  const std::size_t src = 2 * (static_cast<std::size_t>(line) * (n_x + 1) + k);
  const std::size_t dst = 2 * (static_cast<std::size_t>(line) * n_x + k);
  const real_t angle = -pi * k / (2 * n_x);
  const real_t yr = ms[src], yi = ms[src + 1];
  sx[dst] = yr * cos(angle) - yi * sin(angle);
  sx[dst + 1] = 0;
}

__global__ void dct_backward_kernel(real_t* ms, const real_t* sx, const int n_lines,
                                    const int n_x, const real_t pi) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_lines) * (n_x + 1)) return;
  const int k = static_cast<int>(idx % (n_x + 1));
  const int line = static_cast<int>(idx / (n_x + 1));
  const std::size_t dst = 2 * (static_cast<std::size_t>(line) * (n_x + 1) + k);
  if (k == n_x) {  // Nyquist
    ms[dst] = 0;
    ms[dst + 1] = 0;
    return;
  }
  const std::size_t src = 2 * (static_cast<std::size_t>(line) * n_x + k);
  const real_t angle = pi * k / (2 * n_x);
  const real_t x = sx[src];
  ms[dst] = x * cos(angle);
  ms[dst + 1] = x * sin(angle);
}

__global__ void scatter_mirrored_kernel(const field::View p, const real_t* lines,
                                        const real_t scale, const int h, const int n_x,
                                        const int n_y, const int nz_local) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_x) * n_y * nz_local) return;
  const int ii = static_cast<int>(idx % n_x);
  const int kk = static_cast<int>((idx / n_x) % nz_local);
  const int jj = static_cast<int>(idx / (static_cast<long long>(n_x) * nz_local));
  const real_t* line = lines + (static_cast<std::size_t>(jj) * nz_local + kk) * (2 * n_x);
  p(ii + h, jj + h, kk + h) = line[ii] * scale;
}

}  // namespace

void pack_rhs_to_lines(real_t* lines, const field::Field& rhs,
                       const parallel::Subdomain& sub, const int n_x, const int n_y,
                       const int nz_local) {
  const dim3 threads(kTile, kTileRows);
  const dim3 blocks((n_x + kTile - 1) / kTile, (nz_local + kTile - 1) / kTile, n_y);
  pack_rhs_kernel<<<blocks, threads>>>(lines, rhs.view(), halo_width, n_x, n_y, nz_local);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void reorder_x_to_y(real_t* sxy, const real_t* sx, const int n_y, const int nz_local,
                    const int n_mode_x) {
  const dim3 threads(kTile, kTileRows);
  const dim3 blocks((n_mode_x + kTile - 1) / kTile, (n_y + kTile - 1) / kTile, nz_local);
  reorder_x_to_y_kernel<<<blocks, threads>>>(sxy, sx, n_y, nz_local, n_mode_x);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void reorder_y_to_x(real_t* sx, const real_t* sxy, const int n_y, const int nz_local,
                    const int n_mode_x) {
  const dim3 threads(kTile, kTileRows);
  const dim3 blocks((n_y + kTile - 1) / kTile, (n_mode_x + kTile - 1) / kTile, nz_local);
  reorder_y_to_x_kernel<<<blocks, threads>>>(sx, sxy, n_y, nz_local, n_mode_x);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void pack_spectrum_to_bands(real_t* D, const real_t* sxy, const int nz_local,
                            const int n_mode_x, const int n_y,
                            const bool pin_mean_row) {
  const long long n = static_cast<long long>(nz_local) * n_mode_x * n_y;
  pack_bands_kernel<<<blocks_for(n), kThreads>>>(D, sxy, nz_local, n_mode_x, n_y,
                                                 pin_mean_row);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void unpack_bands_to_spectrum(real_t* sxy, const real_t* D, const int nz_local,
                              const int n_mode_x, const int n_y) {
  const long long n = static_cast<long long>(nz_local) * n_mode_x * n_y;
  unpack_bands_kernel<<<blocks_for(n), kThreads>>>(sxy, D, nz_local, n_mode_x, n_y);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void scatter_lines_to_field(field::Field& phi, const real_t* lines, const real_t scale,
                            const parallel::Subdomain& sub, const int n_x,
                            const int n_y, const int nz_local) {
  const dim3 threads(kTile, kTileRows);
  const dim3 blocks((nz_local + kTile - 1) / kTile, (n_x + kTile - 1) / kTile, n_y);
  scatter_lines_kernel<<<blocks, threads>>>(phi.view(), lines, scale, halo_width, n_x, n_y,
                                            nz_local);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void pack_rhs_to_mirrored_lines(real_t* lines, const field::Field& rhs,
                                const parallel::Subdomain& sub, const int n_x,
                                const int n_y, const int nz_local) {
  (void)sub;
  const long long n = static_cast<long long>(n_x) * n_y * nz_local;
  pack_mirrored_kernel<<<blocks_for(n), kThreads>>>(lines, rhs.view(), halo_width, n_x,
                                                    n_y, nz_local);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void dct_twiddle_forward(real_t* sx, const real_t* mirror_spectrum, const int n_lines,
                         const int n_x) {
  const long long n = static_cast<long long>(n_lines) * n_x;
  dct_forward_kernel<<<blocks_for(n), kThreads>>>(sx, mirror_spectrum, n_lines, n_x,
                                                  std::acos(real_t(-1)));
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void dct_twiddle_backward(real_t* mirror_spectrum, const real_t* sx, const int n_lines,
                          const int n_x) {
  const long long n = static_cast<long long>(n_lines) * (n_x + 1);
  dct_backward_kernel<<<blocks_for(n), kThreads>>>(mirror_spectrum, sx, n_lines, n_x,
                                                   std::acos(real_t(-1)));
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void scatter_mirrored_lines_to_field(field::Field& phi, const real_t* lines,
                                     const real_t scale, const parallel::Subdomain& sub,
                                     const int n_x, const int n_y, const int nz_local) {
  (void)sub;
  const long long n = static_cast<long long>(n_x) * n_y * nz_local;
  scatter_mirrored_kernel<<<blocks_for(n), kThreads>>>(phi.view(), lines, scale,
                                                       halo_width, n_x, n_y, nz_local);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

real_t interior_sum(const field::Field& phi, const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  real_t* total = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&total), sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(total, 0, sizeof(real_t)));
  interior_sum_kernel<<<blocks_for(n), kThreads>>>(phi.view(), halo_width, i1, i2, i3,
                                                   total);
  MPMSTD_CUDA_CHECK_LAUNCH();
  real_t host = 0;
  MPMSTD_CUDA_CHECK(cudaMemcpy(&host, total, sizeof(real_t), cudaMemcpyDeviceToHost));
  cudaFree(total);
  return host;
}

void subtract_constant(field::Field& phi, const parallel::Subdomain& sub,
                       const real_t value) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  subtract_kernel<<<blocks_for(n), kThreads>>>(phi.view(), halo_width, i1, i2, i3,
                                               value);
  MPMSTD_CUDA_CHECK_LAUNCH();
}






}  // namespace mpmstd::equation::pressure
