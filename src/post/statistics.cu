#include "post/statistics.hpp"

#include <mpi.h>

#include "common/cuda_check.hpp"

// GPU 백엔드 — statistics.cpp 의 쌍둥이. 프로파일은 디바이스 누적(atomicAdd —
// 합 순서가 런마다 다를 수 있다) 후 소량만 D2H, 스칼라 축소는 블록 부분합 +
// atomic. 호스트로 내려가는 것은 프로파일(nz_global 개)과 스칼라뿐이다
// (2026-10-04 사용자 지시: 통계 출력 때만 CPU 로).

namespace mpmstd::post {

namespace {

enum : int { kThreads = 256 };

inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

__device__ inline void atomic_max_double(real_t* address, const real_t value) {
  unsigned long long* target = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *target;
  while (__longlong_as_double(observed) < value) {
    const unsigned long long assumed = observed;
    observed = atomicCAS(target, assumed, __double_as_longlong(value));
    if (observed == assumed) break;
  }
}

__global__ void profile_kernel(const field::ConstView f, const int h, const int i1,
                               const int i2, const int i3, const int z_offset,
                               const bool squared, real_t* profile) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const real_t value = squared ? f(i, j, k) * f(i, j, k) : f(i, j, k);
  atomicAdd(&profile[z_offset + (k - h)], value);
}

__global__ void product_profile_kernel(const field::ConstView f,
                                       const field::ConstView g, const int h,
                                       const int i1, const int i2, const int i3,
                                       const int z_offset, real_t* profile) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  atomicAdd(&profile[z_offset + (k - h)], f(i, j, k) * g(i, j, k));
}

__global__ void bulk_kernel(const field::ConstView u, const real_t* dx,
                            const real_t* dy, const real_t* dz, const int h,
                            const int i1, const int i2, const int i3,
                            real_t* flux_and_volume) {
  __shared__ real_t flux[kThreads];
  __shared__ real_t volume[kThreads];
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  real_t my_flux = 0, my_volume = 0;
  if (idx < static_cast<long long>(i1) * i2 * i3) {
    const int k = h + static_cast<int>(idx % i3);
    const int j = h + static_cast<int>((idx / i3) % i2);
    const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
    const real_t dv = dx[i] * dy[j] * dz[k];
    my_flux = real_t(0.5) * (u(i, j, k) + u(i + 1, j, k)) * dv;
    my_volume = dv;
  }
  flux[threadIdx.x] = my_flux;
  volume[threadIdx.x] = my_volume;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      flux[threadIdx.x] += flux[threadIdx.x + s];
      volume[threadIdx.x] += volume[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&flux_and_volume[0], flux[0]);
    atomicAdd(&flux_and_volume[1], volume[0]);
  }
}

__global__ void divergence_max_kernel(const field::ConstView u,
                                      const field::ConstView v,
                                      const field::ConstView w, const real_t* dx,
                                      const real_t* dy, const real_t* dz, const int h,
                                      const int i1, const int i2, const int i3,
                                      real_t* worst) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const real_t divergence =
      fabs((u(i + 1, j, k) - u(i, j, k)) / dx[i]
           + (v(i, j + 1, k) - v(i, j, k)) / dy[j]
           + (w(i, j, k + 1) - w(i, j, k)) / dz[k]);
  atomic_max_double(worst, divergence);
}

std::vector<real_t> profile_impl(const field::Field& F, const parallel::Subdomain& sub,
                                 const bool squared) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nz_global = sub.z().n_global;
  real_t* device_profile = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_profile),
                               nz_global * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(device_profile, 0, nz_global * sizeof(real_t)));
  const long long n = static_cast<long long>(i1) * i2 * i3;
  profile_kernel<<<blocks_for(n), kThreads>>>(F.view(), halo_width, i1, i2, i3,
                                              sub.z().offset, squared, device_profile);
  MPMSTD_CUDA_CHECK_LAUNCH();
  std::vector<real_t> profile(nz_global);
  MPMSTD_CUDA_CHECK(cudaMemcpy(profile.data(), device_profile,
                               nz_global * sizeof(real_t), cudaMemcpyDeviceToHost));
  cudaFree(device_profile);
  MPI_Allreduce(MPI_IN_PLACE, profile.data(), nz_global, MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
  const real_t plane_count =
      static_cast<real_t>(sub.x().n_global) * sub.y().n_global;  // 전역! (16배 버그)
  for (real_t& v : profile) v /= plane_count;
  return profile;
}

}  // namespace

std::vector<real_t> plane_average(const field::Field& F,
                                  const parallel::Subdomain& sub) {
  return profile_impl(F, sub, /*squared=*/false);
}

std::vector<real_t> plane_average_square(const field::Field& F,
                                         const parallel::Subdomain& sub) {
  return profile_impl(F, sub, /*squared=*/true);
}

std::vector<real_t> plane_average_product(const field::Field& F,
                                          const field::Field& G,
                                          const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nz_global = sub.z().n_global;
  real_t* device_profile = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_profile),
                               nz_global * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(device_profile, 0, nz_global * sizeof(real_t)));
  const long long n = static_cast<long long>(i1) * i2 * i3;
  product_profile_kernel<<<blocks_for(n), kThreads>>>(
      F.view(), G.view(), halo_width, i1, i2, i3, sub.z().offset, device_profile);
  MPMSTD_CUDA_CHECK_LAUNCH();
  std::vector<real_t> profile(nz_global);
  MPMSTD_CUDA_CHECK(cudaMemcpy(profile.data(), device_profile,
                               nz_global * sizeof(real_t), cudaMemcpyDeviceToHost));
  cudaFree(device_profile);
  MPI_Allreduce(MPI_IN_PLACE, profile.data(), nz_global, MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
  const real_t plane_count =
      static_cast<real_t>(sub.x().n_global) * sub.y().n_global;
  for (real_t& v : profile) v /= plane_count;
  return profile;
}

real_t bulk_velocity(const field::Field& U, const grid::Grid& grid,
                     const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  real_t* device_pair = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_pair),
                               2 * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(device_pair, 0, 2 * sizeof(real_t)));
  const long long n = static_cast<long long>(i1) * i2 * i3;
  bulk_kernel<<<blocks_for(n), kThreads>>>(
      U.view(), grid.x.device_cell_width.data(), grid.y.device_cell_width.data(),
      grid.z.device_cell_width.data(), halo_width, i1, i2, i3, device_pair);
  MPMSTD_CUDA_CHECK_LAUNCH();
  real_t local[2] = {0, 0};
  MPMSTD_CUDA_CHECK(cudaMemcpy(local, device_pair, 2 * sizeof(real_t),
                               cudaMemcpyDeviceToHost));
  cudaFree(device_pair);
  real_t total[2] = {0, 0};
  MPI_Allreduce(local, total, 2, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  return (total[1] > 0) ? total[0] / total[1] : real_t(0);
}

real_t max_divergence(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  real_t* device_worst = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_worst), sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(device_worst, 0, sizeof(real_t)));
  const long long n = static_cast<long long>(i1) * i2 * i3;
  divergence_max_kernel<<<blocks_for(n), kThreads>>>(
      U.view(), V.view(), W.view(), grid.x.device_cell_width.data(),
      grid.y.device_cell_width.data(), grid.z.device_cell_width.data(), halo_width,
      i1, i2, i3, device_worst);
  MPMSTD_CUDA_CHECK_LAUNCH();
  real_t worst = 0;
  MPMSTD_CUDA_CHECK(cudaMemcpy(&worst, device_worst, sizeof(real_t),
                               cudaMemcpyDeviceToHost));
  cudaFree(device_worst);
  real_t global = 0;
  MPI_Allreduce(&worst, &global, 1, MPI_DOUBLE, MPI_MAX, sub.topology().cart_comm());
  return global;
}

}  // namespace mpmstd::post
