#include "physics/channel_forcing.hpp"

#include "post/statistics.hpp"

namespace mpmstd::physics {

void add_mean_pressure_gradient(field::Field& rhs_u, const parallel::Subdomain& sub,
                                const real_t mean_dPdx, const real_t dt) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const real_t source = -dt * mean_dPdx;
  const field::View r = rhs_u.view();
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) r(i, j, k) += source;
}

void correct_bulk_velocity(field::Field& U, const grid::Grid& grid,
                           const parallel::Subdomain& sub, const real_t target_bulk,
                           const real_t dt, real_t& mean_dPdx) {
  const int h = halo_width;
  const real_t bulk = post::bulk_velocity(U, grid, sub);
  const real_t gradient_increment = (dt > real_t(1e-15)) ? (bulk - target_bulk) / dt
                                                         : real_t(0);
  const real_t shift = -dt * gradient_increment;  // = target − bulk
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const field::View u = U.view();
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) u(i, j, k) += shift;
  mean_dPdx += gradient_increment;
}

}  // namespace mpmstd::physics
