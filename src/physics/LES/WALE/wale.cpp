#include "physics/LES/WALE/wale.hpp"

#include "common/field_checks.hpp"
#include "physics/LES/WALE/wale_stencil.hpp"

// CPU 백엔드 — 식은 wale_stencil.hpp 한 곳 (GPU 쌍둥이 wale.cu).

namespace mpmstd::physics::les {

void calculate_eddy_viscosity_wale(field::Field& mu_hat, const field::Field& U,
                                   const field::Field& V, const field::Field& W,
                                   const grid::Grid& grid, const parallel::Subdomain& sub,
                                   const real_t nu, const real_t c_wale) {
  check_field_matches("calculate_eddy_viscosity_wale", mu_hat, sub);
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  real_t* m = mu_hat.data();
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t nu_t = stencil::eddy_viscosity_cell_wale(
            i, j, k, ny, nz, U.data(), V.data(), W.data(), grid.x.cell_width.data(),
            grid.x.center_distance.data(), grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), c_wale);
        m[stencil::idx3(i, j, k, ny, nz)] = 1 + nu_t / nu;
      }
}

}  // namespace mpmstd::physics::les
