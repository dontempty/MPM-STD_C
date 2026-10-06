#include "physics/LES/constant_vreman/vreman.hpp"

#include <stdexcept>

#include "boundary/wall.hpp"
#include "common/field_checks.hpp"
#include "physics/LES/constant_vreman/vreman_stencil.hpp"
#include "physics/LES/les.hpp"

// CPU 백엔드 — 식은 vreman_stencil.hpp 한 곳 (GPU 쌍둥이 vreman.cu).

namespace mpmstd::physics::les {

void calculate_eddy_viscosity_constant_vreman(field::Field& mu_hat, const field::Field& U,
                                              const field::Field& V, const field::Field& W,
                                              const grid::Grid& grid, const parallel::Subdomain& sub,
                                              const real_t nu, const real_t c_vreman) {
  check_field_matches("calculate_eddy_viscosity_constant_vreman", mu_hat, sub);
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  real_t* m = mu_hat.data();
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t nu_t = stencil::eddy_viscosity_cell(
            i, j, k, ny, nz, U.data(), V.data(), W.data(), grid.x.cell_width.data(),
            grid.x.center_distance.data(), grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), c_vreman);
        m[stencil::idx3(i, j, k, ny, nz)] = 1 + nu_t / nu;
      }
}

void synchronize_viscosity_ratio(field::Field& mu_hat, const parallel::Subdomain& sub) {
  sub.exchange_halo(mu_hat.data());
  for (int axis = 0; axis < 3; ++axis) {
    if (sub.topology().axis(axis).periodic) continue;
    boundary::fill_wall_value(mu_hat, sub, axis, 0, 1.0);  // 벽 ghost: ν_t = 0 → μ̂ = 1
    boundary::fill_wall_value(mu_hat, sub, axis, 1, 1.0);
  }
}

}  // namespace mpmstd::physics::les
