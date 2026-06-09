#include "driver/cfl.hpp"
#include "common/macros.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>

// P1 — convective CFL dt (faithful port of channel max_cfl_impl).

namespace mpmstd::driver {

real_t compute_cfl_dt_cpu(const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                          const core::Grid& grid, const core::Subdomain& sub, real_t max_cfl, real_t dt_cap) {
  const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = U.data(); const real_t* v = V.data(); const real_t* w = W.data();
  const real_t* dx1 = grid.dx_ptr(Direction::X);
  const real_t* dx2 = grid.dx_ptr(Direction::Y);
  const real_t* dx3 = grid.dx_ptr(Direction::Z);

  double local = 1e-30;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const double c = std::fabs(double(u[(i * n2 + j) * n3 + k])) / double(dx1[i])
                       + std::fabs(double(v[(i * n2 + j) * n3 + k])) / double(dx2[j])
                       + std::fabs(double(w[(i * n2 + j) * n3 + k])) / double(dx3[k]);
        if (c > local) local = c;
      }
  double speed = local;
  MPI_Allreduce(&local, &speed, 1, MPI_DOUBLE, MPI_MAX, sub.topology().cart_comm());

  if (speed > 1e-14)
    return static_cast<real_t>(std::min(double(max_cfl) / speed, double(dt_cap)));
  return dt_cap;
}

} // namespace mpmstd::driver
