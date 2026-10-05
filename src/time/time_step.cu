#include "time/time_step.hpp"

#include <algorithm>

#include <mpi.h>

#include "common/cuda_check.hpp"

// GPU 백엔드 — time_step.cpp 의 쌍둥이 (CFL max 축소).

namespace mpmstd::time_integration {

namespace {

enum : int { kThreads = 256 };

__device__ inline void atomic_max_double(real_t* address, const real_t value) {
  unsigned long long* target = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *target;
  while (__longlong_as_double(observed) < value) {
    const unsigned long long assumed = observed;
    observed = atomicCAS(target, assumed, __double_as_longlong(value));
    if (observed == assumed) break;
  }
}

__global__ void speed_max_kernel(const field::ConstView u, const field::ConstView v,
                                 const field::ConstView w, const real_t* dx,
                                 const real_t* dy, const real_t* dz, const int h,
                                 const int i1, const int i2, const int i3,
                                 real_t* worst) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const real_t speed = fabs(u(i, j, k)) / dx[i] + fabs(v(i, j, k)) / dy[j]
                       + fabs(w(i, j, k)) / dz[k];
  atomic_max_double(worst, speed);
}

}  // namespace

real_t compute_cfl_dt(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub, const real_t max_cfl,
                      const real_t dt_cap) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  real_t* device_worst = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_worst), sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(device_worst, 0, sizeof(real_t)));
  const long long n = static_cast<long long>(i1) * i2 * i3;
  speed_max_kernel<<<static_cast<int>((n + kThreads - 1) / kThreads), kThreads>>>(
      U.view(), V.view(), W.view(), grid.x.device_cell_width.data(),
      grid.y.device_cell_width.data(), grid.z.device_cell_width.data(), halo_width,
      i1, i2, i3, device_worst);
  MPMSTD_CUDA_CHECK_LAUNCH();
  real_t local_max = 0;
  MPMSTD_CUDA_CHECK(cudaMemcpy(&local_max, device_worst, sizeof(real_t),
                               cudaMemcpyDeviceToHost));
  cudaFree(device_worst);
  real_t global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                sub.topology().cart_comm());
  const real_t dt_cfl = (global_max > 0) ? max_cfl / global_max : dt_cap;
  return std::min(dt_cfl, dt_cap);
}

}  // namespace mpmstd::time_integration
