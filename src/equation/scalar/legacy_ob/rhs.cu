#include "equation/scalar/legacy_ob/rhs.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/transport_stencil.hpp"

// GPU 백엔드 — rhs.cpp 의 쌍둥이 (수식은 transport_stencil.hpp 공용).

namespace mpmstd::equation::scalar {

namespace {

enum : int { kThreads = 256 };

__global__ void diffusion_kernel(const field::View rhs, const field::ConstView T,
                                 const real_t* dx_x, const real_t* dmx_x,
                                 const real_t* dx_y, const real_t* dmx_y,
                                 const real_t* dx_z, const real_t* dmx_z, const int h,
                                 const int i1, const int i2, const int i3,
                                 const real_t dt_alpha) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  rhs(i, j, k) += dt_alpha * equation::stencil::diffusion(
                      T, i, j, k, dx_x, dmx_x, dx_y, dmx_y, dx_z, dmx_z, false);
}

__global__ void convection_kernel(const field::View rhs, const field::ConstView T,
                                  const field::ConstView U, const field::ConstView V,
                                  const field::ConstView W, const real_t* dx_x,
                                  const real_t* dx_y, const real_t* dx_z, const int h,
                                  const int i1, const int i2, const int i3,
                                  const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  rhs(i, j, k) -= dt * equation::stencil::convection_divergence(T, U, V, W, i, j, k,
                                                                dx_x, dx_y, dx_z);
}

inline int blocks_for(const parallel::Subdomain& sub) {
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

}  // namespace

void add_diffusion(field::Field& rhs, const field::Field& T, const grid::Grid& grid,
                   const parallel::Subdomain& sub, const real_t alpha,
                   const real_t dt) {
  check_field_matches("add_diffusion", rhs, sub);
  check_field_matches("add_diffusion", T, sub);
  diffusion_kernel<<<blocks_for(sub), kThreads>>>(
      rhs.view(), T.view(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, sub.x().n_local,
      sub.y().n_local, sub.z().n_local, dt * alpha);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_convection(field::Field& rhs, const field::Field& T, const field::Field& U,
                    const field::Field& V, const field::Field& W,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t dt) {
  check_field_matches("add_convection", rhs, sub);
  check_field_matches("add_convection", T, sub);
  check_field_matches("add_convection", U, sub);
  check_field_matches("add_convection", V, sub);
  check_field_matches("add_convection", W, sub);
  convection_kernel<<<blocks_for(sub), kThreads>>>(
      rhs.view(), T.view(), U.view(), V.view(), W.view(),
      grid.x.device_cell_width.data(), grid.y.device_cell_width.data(),
      grid.z.device_cell_width.data(), halo_width, sub.x().n_local,
      sub.y().n_local, sub.z().n_local, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::scalar
