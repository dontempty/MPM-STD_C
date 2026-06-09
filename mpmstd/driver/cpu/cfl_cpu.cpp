#include "driver/cfl.hpp"
#include "common/macros.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>

namespace mpmstd::driver {

real_t compute_cfl_dt_cpu(const core::Domain& domain, const core::CpuFields& fields, real_t max_cfl, real_t dt_cap) {
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = fields[core::Var::U].data();
  const real_t* v = fields[core::Var::V].data();
  const real_t* w = fields[core::Var::W].data();
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z);

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
  MPI_Allreduce(&local, &speed, 1, MPI_DOUBLE, MPI_MAX, domain.sub.topology().cart_comm());

  if (speed > 1e-14)
    return static_cast<real_t>(std::min(double(max_cfl) / speed, double(dt_cap)));
  return dt_cap;
}

} // namespace mpmstd::driver
