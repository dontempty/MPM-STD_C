#include "physics/forcing/forcing.hpp"
#include "common/macros.hpp"

#include <mpi.h>

// P1 channel forcing rewired to Domain + Fields. dP/dx threaded by the caller.

namespace mpmstd::physics {

void apply_body_force_cpu(core::CpuFields& fields, real_t dpdx, real_t dt) {
  core::CpuField& U = fields[core::Var::U];
  const auto nt = U.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  real_t* u = U.data();
  const real_t val = static_cast<real_t>(-dt * dpdx);
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k)
        u[(i * n2 + j) * n3 + k] += val;
}

double channel_total_volume_cpu(const core::Domain& domain) {
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z);
  double local = 0.0;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k)
        local += double(dx1[i]) * double(dx2[j]) * double(dx3[k]);
  double total = 0.0;
  MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, domain.sub.topology().cart_comm());
  return total;
}

double channel_bulk_velocity_cpu(const core::Domain& domain, const core::CpuFields& fields, double total_vol) {
  const core::CpuField& U = fields[core::Var::U];
  const auto nt = U.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = U.data();
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z);
  double local = 0.0;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const double uc = 0.5 * (double(u[(i * n2 + j) * n3 + k]) + double(u[((i + 1) * n2 + j) * n3 + k]));
        local += uc * double(dx1[i]) * double(dx2[j]) * double(dx3[k]);
      }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, domain.sub.topology().cart_comm());
  return global / total_vol;
}

double apply_mass_flow_correction_cpu(const core::Domain& domain, core::CpuFields& fields,
                                      double Ub_target, double dt, double total_vol, double& dpdx) {
  const double Ub      = channel_bulk_velocity_cpu(domain, fields, total_vol);
  const double DMpresg = (dt > 1.0e-15) ? (Ub - Ub_target) / dt : 0.0;
  const real_t shift   = static_cast<real_t>(-dt * DMpresg);

  core::CpuField& U = fields[core::Var::U];
  const auto nt = U.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  real_t* u = U.data();
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k)
        u[(i * n2 + j) * n3 + k] += shift;

  dpdx += DMpresg;
  return dpdx;
}

} // namespace mpmstd::physics
