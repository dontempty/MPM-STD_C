#include "time/time_step.hpp"

#include <algorithm>
#include <cmath>

#include <mpi.h>


namespace mpmstd::time_integration {

real_t compute_cfl_dt(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub, const real_t max_cfl,
                      const real_t dt_cap) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const field::ConstView u = U.view();
  const field::ConstView v = V.view();
  const field::ConstView w = W.view();
  const real_t* dx = grid.x.cell_width.data();
  const real_t* dy = grid.y.cell_width.data();
  const real_t* dz = grid.z.cell_width.data();
  real_t local_max = 0;
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t speed = std::abs(u(i, j, k)) / dx[i]
                             + std::abs(v(i, j, k)) / dy[j]
                             + std::abs(w(i, j, k)) / dz[k];
        if (speed > local_max) local_max = speed;
      }
  real_t global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                sub.topology().cart_comm());
  const real_t dt_cfl = (global_max > 0) ? max_cfl / global_max : dt_cap;
  return std::min(dt_cfl, dt_cap);
}

}  // namespace mpmstd::time_integration
