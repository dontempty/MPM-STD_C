#include "physics/channel_forcing.hpp"

#include "common/cuda_check.hpp"
#include "post/statistics.hpp"

// GPU 백엔드 — channel_forcing.cpp 의 쌍둥이 (벌크 측정은 post::bulk_velocity).

namespace mpmstd::physics {

namespace {

enum : int { kThreads = 256 };

__global__ void add_uniform_kernel(const field::View f, const int h, const int i1,
                                   const int i2, const int i3, const real_t value) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  f(i, j, k) += value;
}

inline void add_uniform(field::Field& f, const parallel::Subdomain& sub,
                        const real_t value) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  add_uniform_kernel<<<static_cast<int>((n + kThreads - 1) / kThreads), kThreads>>>(
      f.view(), halo_width, i1, i2, i3, value);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void add_mean_pressure_gradient(field::Field& rhs_u, const parallel::Subdomain& sub,
                                const real_t mean_dPdx, const real_t dt) {
  add_uniform(rhs_u, sub, -dt * mean_dPdx);
}

void correct_bulk_velocity(field::Field& U, const grid::Grid& grid,
                           const parallel::Subdomain& sub, const real_t target_bulk,
                           const real_t dt, real_t& mean_dPdx) {
  const real_t bulk = post::bulk_velocity(U, grid, sub);
  const real_t gradient_increment = (dt > real_t(1e-15)) ? (bulk - target_bulk) / dt
                                                         : real_t(0);
  add_uniform(U, sub, -dt * gradient_increment);  // = target − bulk
  mean_dPdx += gradient_increment;
}

}  // namespace mpmstd::physics
