#include "parallel/subdomain/subdomain.hpp"

#include <mpi.h>

#include "common/cuda_check.hpp"

// GPU halo 교환 — CPU 경로(subdomain.cpp)와 같은 이웃·태그·순서(x→y→z)를
// 따르되, 벌어진 y/z 평면은 "팩 커널 → 연속 디바이스 버퍼 → MPI(CUDA-aware)"
// 로 보낸다. 디바이스 포인터에 subarray datatype 을 물리면 UCX 가 segfault
// 하는 것을 실측해서(2026-10-04) MPM-STD_C 와 같은 팩 방식을 쓴다.
// 벽(MPI_PROC_NULL)에서는 Sendrecv 가 no-op 이므로 **unpack 도 건너뛴다** —
// 안 그러면 스크래치 버퍼의 낡은 값이 ghost 를 덮는다 (CPU 경로의 "벽 ghost
// 불변" 계약과 동일하게 유지).

namespace mpmstd::parallel {

namespace {

enum : int { kThreads = 256 };

inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

// y-평면 (j 고정): buffer[i*nz + k] ↔ f[(i*ny + j)*nz + k]
__global__ void pack_y_kernel(real_t* buffer, const real_t* f, const int j_fixed,
                              const int nx, const int ny, const int nz) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= static_cast<long long>(nx) * nz) return;
  const int i = static_cast<int>(q / nz);
  const int k = static_cast<int>(q % nz);
  buffer[q] = f[(static_cast<std::size_t>(i) * ny + j_fixed) * nz + k];
}

__global__ void unpack_y_kernel(real_t* f, const real_t* buffer, const int j_fixed,
                                const int nx, const int ny, const int nz) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= static_cast<long long>(nx) * nz) return;
  const int i = static_cast<int>(q / nz);
  const int k = static_cast<int>(q % nz);
  f[(static_cast<std::size_t>(i) * ny + j_fixed) * nz + k] = buffer[q];
}

// z-평면 (k 고정): buffer[i*ny + j] ↔ f[(i*ny + j)*nz + k]
__global__ void pack_z_kernel(real_t* buffer, const real_t* f, const int k_fixed,
                              const int nx, const int ny, const int nz) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= static_cast<long long>(nx) * ny) return;
  buffer[q] = f[q * nz + k_fixed];
}

__global__ void unpack_z_kernel(real_t* f, const real_t* buffer, const int k_fixed,
                                const int nx, const int ny, const int nz) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= static_cast<long long>(nx) * ny) return;
  f[q * nz + k_fixed] = buffer[q];
}

}  // namespace

void Subdomain::exchange_halo_device(real_t* f) const {
  const int nx = x_.n_total;
  const int ny = y_.n_total;
  const int nz = z_.n_total;
  const int h = halo_width;

  // 앞선 커널들이 f 에 쓴 것을 완료시킨다 (MPI 는 스트림을 모른다).
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());

  // ── x 방향 — 평면이 연속이라 필드에서 바로 보낸다 (CPU 와 동일 태그) ──
  // [최적화 2026-10-17] 축 분할 1 + 주기 → 이웃이 자기 자신. MPI_Sendrecv(self) 는 UCX 호스트 경유라 np 에 따라 급격히 느려짐
  //   (0.5 MB 면: np 1 0.1 → np 8 6 ms). 자기 자신이면 디바이스 복사로 랩을 채움(비트 동일). weak np 8 효율 57 → 86 %.
  {
    const int plane = ny * nz;
    const bool self = topo_.x().plus_neighbor == topo_.x().rank
                      && topo_.x().minus_neighbor == topo_.x().rank;
    if (self) {
      MPMSTD_CUDA_CHECK(cudaMemcpy(f + static_cast<std::size_t>(nx - h) * plane,
                                   f + static_cast<std::size_t>(h) * plane,
                                   static_cast<std::size_t>(plane) * sizeof(real_t),
                                   cudaMemcpyDeviceToDevice));
      MPMSTD_CUDA_CHECK(cudaMemcpy(f, f + static_cast<std::size_t>(nx - 2 * h) * plane,
                                   static_cast<std::size_t>(plane) * sizeof(real_t),
                                   cudaMemcpyDeviceToDevice));
    } else {
      MPI_Sendrecv(f + static_cast<std::size_t>(h) * plane, plane, MPI_DOUBLE,
                   topo_.x().minus_neighbor, 100,
                   f + static_cast<std::size_t>(nx - h) * plane, plane, MPI_DOUBLE,
                   topo_.x().plus_neighbor, 100, topo_.x().comm, MPI_STATUS_IGNORE);
      MPI_Sendrecv(f + static_cast<std::size_t>(nx - 2 * h) * plane, plane, MPI_DOUBLE,
                   topo_.x().plus_neighbor, 101, f, plane, MPI_DOUBLE,
                   topo_.x().minus_neighbor, 101, topo_.x().comm, MPI_STATUS_IGNORE);
    }
  }

  // ── y 방향 — 팩 → 연속 버퍼 MPI → 언팩 ──
  {
    const long long face = static_cast<long long>(nx) * nz;
    pack_y_kernel<<<blocks_for(face), kThreads>>>(y_send_minus_.data(), f, h, nx, ny, nz);
    pack_y_kernel<<<blocks_for(face), kThreads>>>(y_send_plus_.data(), f, ny - 2 * h,
                                                  nx, ny, nz);
    MPMSTD_CUDA_CHECK_LAUNCH();
    const bool self = topo_.y().plus_neighbor == topo_.y().rank
                      && topo_.y().minus_neighbor == topo_.y().rank;
    if (self) {
      // 자기 자신 랩: 팩한 버퍼를 그대로 반대쪽 ghost 에 언팩 (x 와 같은 이유로 MPI 생략)
      unpack_y_kernel<<<blocks_for(face), kThreads>>>(f, y_send_minus_.data(), ny - h, nx, ny, nz);
      unpack_y_kernel<<<blocks_for(face), kThreads>>>(f, y_send_plus_.data(), 0, nx, ny, nz);
      MPMSTD_CUDA_CHECK_LAUNCH();
    } else {
      MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Sendrecv(y_send_minus_.data(), static_cast<int>(face), MPI_DOUBLE,
                   topo_.y().minus_neighbor, 200, y_recv_plus_.data(),
                   static_cast<int>(face), MPI_DOUBLE, topo_.y().plus_neighbor, 200,
                   topo_.y().comm, MPI_STATUS_IGNORE);
      MPI_Sendrecv(y_send_plus_.data(), static_cast<int>(face), MPI_DOUBLE,
                   topo_.y().plus_neighbor, 201, y_recv_minus_.data(),
                   static_cast<int>(face), MPI_DOUBLE, topo_.y().minus_neighbor, 201,
                   topo_.y().comm, MPI_STATUS_IGNORE);
      if (topo_.y().plus_neighbor != MPI_PROC_NULL) {
        unpack_y_kernel<<<blocks_for(face), kThreads>>>(f, y_recv_plus_.data(),
                                                        ny - h, nx, ny, nz);
        MPMSTD_CUDA_CHECK_LAUNCH();
      }
      if (topo_.y().minus_neighbor != MPI_PROC_NULL) {
        unpack_y_kernel<<<blocks_for(face), kThreads>>>(f, y_recv_minus_.data(), 0, nx,
                                                        ny, nz);
        MPMSTD_CUDA_CHECK_LAUNCH();
      }
    }
  }

  // ── z 방향 — y 언팩이 끝난 평면을 담도록 완료 후 팩 (CPU 순서와 동일) ──
  {
    const long long face = static_cast<long long>(nx) * ny;
    MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
    pack_z_kernel<<<blocks_for(face), kThreads>>>(z_send_minus_.data(), f, h, nx, ny, nz);
    pack_z_kernel<<<blocks_for(face), kThreads>>>(z_send_plus_.data(), f, nz - 2 * h,
                                                  nx, ny, nz);
    MPMSTD_CUDA_CHECK_LAUNCH();
    MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Sendrecv(z_send_minus_.data(), static_cast<int>(face), MPI_DOUBLE,
                 topo_.z().minus_neighbor, 300, z_recv_plus_.data(),
                 static_cast<int>(face), MPI_DOUBLE, topo_.z().plus_neighbor, 300,
                 topo_.z().comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(z_send_plus_.data(), static_cast<int>(face), MPI_DOUBLE,
                 topo_.z().plus_neighbor, 301, z_recv_minus_.data(),
                 static_cast<int>(face), MPI_DOUBLE, topo_.z().minus_neighbor, 301,
                 topo_.z().comm, MPI_STATUS_IGNORE);
    if (topo_.z().plus_neighbor != MPI_PROC_NULL) {
      unpack_z_kernel<<<blocks_for(face), kThreads>>>(f, z_recv_plus_.data(), nz - h,
                                                      nx, ny, nz);
      MPMSTD_CUDA_CHECK_LAUNCH();
    }
    if (topo_.z().minus_neighbor != MPI_PROC_NULL) {
      unpack_z_kernel<<<blocks_for(face), kThreads>>>(f, z_recv_minus_.data(), 0, nx,
                                                      ny, nz);
      MPMSTD_CUDA_CHECK_LAUNCH();
    }
  }
}

}  // namespace mpmstd::parallel
