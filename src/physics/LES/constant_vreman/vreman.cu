#include "physics/LES/constant_vreman/vreman.hpp"

#include "boundary/wall.hpp"
#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "physics/LES/constant_vreman/vreman_stencil.hpp"
#include "physics/LES/les.hpp"

// GPU 백엔드 — vreman.cpp 의 쌍둥이 (식은 vreman_stencil.hpp).

namespace mpmstd::physics::les {

namespace {

__global__ void vreman_kernel(real_t* m, const real_t* U, const real_t* V, const real_t* W,
                              const real_t* dx_x, const real_t* dmx_x, const real_t* dx_y,
                              const real_t* dmx_y, const real_t* dx_z, const real_t* dmx_z,
                              const int h, const int i1, const int i2, const int i3,
                              const int ny_total, const int nz_total, const real_t nu,
                              const real_t c_vreman) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int kk = static_cast<int>(idx % i3);
  const int jj = static_cast<int>((idx / i3) % i2);
  const int ii = static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const int i = ii + h, j = jj + h, k = kk + h;
  const real_t nu_t = stencil::eddy_viscosity_cell(i, j, k, ny_total, nz_total, U, V, W, dx_x,
                                                   dmx_x, dx_y, dmx_y, dx_z, dmx_z, c_vreman);
  m[stencil::idx3(i, j, k, ny_total, nz_total)] = 1 + nu_t / nu;
}

}  // namespace

void calculate_eddy_viscosity_constant_vreman(field::Field& mu_hat, const field::Field& U,
                                              const field::Field& V, const field::Field& W,
                                              const grid::Grid& grid, const parallel::Subdomain& sub,
                                              const real_t nu, const real_t c_vreman) {
  check_field_matches("calculate_eddy_viscosity_constant_vreman", mu_hat, sub);
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  const int threads = 256;
  vreman_kernel<<<static_cast<int>((n + threads - 1) / threads), threads>>>(
      mu_hat.data(), U.data(), V.data(), W.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), h, i1, i2, i3, sub.y().n_total, sub.z().n_total, nu,
      c_vreman);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void synchronize_viscosity_ratio(field::Field& mu_hat, const parallel::Subdomain& sub) {
  sub.exchange_halo(mu_hat.data());
  for (int axis = 0; axis < 3; ++axis) {
    if (sub.topology().axis(axis).periodic) continue;
    boundary::fill_wall_value(mu_hat, sub, axis, 0, 1.0);
    boundary::fill_wall_value(mu_hat, sub, axis, 1, 1.0);
  }
}

}  // namespace mpmstd::physics::les
