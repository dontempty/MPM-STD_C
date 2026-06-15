#include "driver/cfl.hpp"
#include "common/macros.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>

namespace mpmstd::driver {

real_t compute_cfl_dt_cpu(const core::Domain& domain, const core::CpuFields& fields, real_t max_cfl, real_t dt_cap) {
  const auto nt = domain.sub.n_total();
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = fields[core::Var::U].data();
  const real_t* v = fields[core::Var::V].data();
  const real_t* w = fields[core::Var::W].data();
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z);

  real_t local_max = real_t{0};
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int p = (i * n2 + j) * n3 + k;
        const real_t speed = std::abs(u[p]) / dx1[i]
                           + std::abs(v[p]) / dx2[j]
                           + std::abs(w[p]) / dx3[k];
        if (speed > local_max) local_max = speed;
      }

  double global_max = double(local_max);
  MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                domain.sub.topology().cart_comm());

  const real_t dt_cfl = (global_max > 0.0) ? real_t(double(max_cfl) / global_max) : dt_cap;
  // CFL-limited, capped at dt_cap. Convection is explicit (OB) so dt MUST follow
  // the advective CFL — pinning dt at dt_cap blows up once velocities grow. For a
  // large fixed dt, make convection implicit instead (see Report 18), not a floor here.
  return std::min(dt_cfl, dt_cap);
}

} // namespace mpmstd::driver
