#include "equation/momentum/solve_kernels.hpp"

#include <cstddef>

#include "common/cuda_check.hpp"
#include "equation/momentum/momentum_stencil.hpp"

// GPU 백엔드 — solve_kernels.cpp 의 쌍둥이. 커플링 수식은 momentum_stencil.hpp.

namespace mpmstd::equation::momentum {

namespace {

enum : int { kThreads = 256 };

__device__ inline std::size_t at(const int i, const int j, const int k,
                                 const int ny_total, const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

struct InteriorIndex {
  int i, j, k;
  bool valid;
};

__device__ inline InteriorIndex decode(const long long idx, const int h, const int i1,
                                       const int i2, const int i3) {
  InteriorIndex out{0, 0, 0, false};
  if (idx >= static_cast<long long>(i1) * i2 * i3) return out;
  out.k = h + static_cast<int>(idx % i3);
  out.j = h + static_cast<int>((idx / i3) % i2);
  out.i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  out.valid = true;
  return out;
}

__global__ void couple_dv_kernel(const field::View dV, const field::ConstView dW,
                                 const field::ConstView V, const real_t* dy,
                                 const real_t* dmy, const real_t* dmz, const int h,
                                 const int i1, const int i2, const int i3,
                                 const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const InteriorIndex c = decode(idx, h, i1, i2, i3);
  if (!c.valid) return;
  dV(c.i, c.j, c.k) -=
      dt * stencil::couple_dv_from_dw(dW, V, c.i, c.j, c.k, dy, dmy, dmz);
}

__global__ void couple_du_kernel(const field::View dU, const field::ConstView dV,
                                 const field::ConstView dW, const field::ConstView U,
                                 const real_t* dx, const real_t* dmx,
                                 const real_t* dmy, const real_t* dmz, const int h,
                                 const int i1, const int i2, const int i3,
                                 const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const InteriorIndex c = decode(idx, h, i1, i2, i3);
  if (!c.valid) return;
  dU(c.i, c.j, c.k) -=
      dt * stencil::couple_du_from_dv_dw(dV, dW, U, c.i, c.j, c.k, dx, dmx, dmy, dmz);
}

inline int blocks_for(const parallel::Subdomain& sub) {
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

}  // namespace

void block_couple_delta_v(field::Field& dV, const field::Field& dW,
                          const field::Field& V, const grid::Grid& grid,
                          const parallel::Subdomain& sub, const real_t dt) {
  couple_dv_kernel<<<blocks_for(sub), kThreads>>>(
      dV.view(), dW.view(), V.view(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_center_distance.data(),
      halo_width, sub.x().n_local, sub.y().n_local, sub.z().n_local, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void block_couple_delta_u(field::Field& dU, const field::Field& dV,
                          const field::Field& dW, const field::Field& U,
                          const grid::Grid& grid, const parallel::Subdomain& sub,
                          const real_t dt) {
  couple_du_kernel<<<blocks_for(sub), kThreads>>>(
      dU.view(), dV.view(), dW.view(), U.view(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_center_distance.data(),
      grid.z.device_center_distance.data(), halo_width, sub.x().n_local,
      sub.y().n_local, sub.z().n_local, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::momentum
