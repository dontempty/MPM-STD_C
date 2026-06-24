#include "equation/pressure/solve.hpp"
#include "equation/pressure/pressure_engine.hpp"   // PressureSolver (complete type)
#include "common/direction.hpp"
#include "common/macros.hpp"

#include <cstddef>
#include <memory>

// P1 pressure step rewired to Domain + Fields. Lazily builds the validated
// pencil-FFT engine (from Domain+BC) and runs the full step on the Fields.

namespace mpmstd::equation {

void add_pressure_gradient_cpu(const core::Domain& domain, core::CpuMomentumSystem& mom,
                               const core::CpuField& P_acc, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* p    = P_acc.data();
  const real_t* dmx1 = domain.grid.dmx_ptr(Direction::X);
  const real_t* dmx2 = domain.grid.dmx_ptr(Direction::Y);
  const real_t* dmx3 = domain.grid.dmx_ptr(Direction::Z);
  real_t* du = mom.rhs_u.data();
  real_t* dv = mom.rhs_v.data();
  real_t* dw = mom.rhs_w.data();
  // dU/dV/dW -= dt·∇P_acc  (same staggered-face gradient as project_; ref mom.f90:344-345)
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
        du[c] -= dt * (p[c] - p[((i-1)*n2+j)*n3+k]) / dmx1[i];
        dv[c] -= dt * (p[c] - p[(i*n2+j-1)*n3+k])   / dmx2[j];
        dw[c] -= dt * (p[c] - p[(i*n2+j)*n3+k-1])   / dmx3[k];
      }
}

void accumulate_pressure_cpu(core::CpuField& P_acc, const core::CpuField& dP) {
  real_t*       pa = P_acc.data();
  const real_t* dp = dP.data();
  const std::size_t n = P_acc.n_elements();
  for (std::size_t q = 0; q < n; ++q) pa[q] += dp[q];   // ref mom.f90... pre.f90:818 P=P+dP
}

void solve_pressure_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::CpuFields& fields, core::PressureSystem& poi, real_t dt,
                        const real_t* inv_rho) {
  if (!poi.engine)
    poi.engine = std::make_shared<pressure::PressureSolver>(domain.grid, domain.sub, bc, domain.tdma);
  if (inv_rho) {
    if (!poi.dPhat)
      poi.dPhat = std::make_unique<core::CpuField>(domain.sub, "dPhat");  // zero-init
    if (!poi.dP0)
      poi.dP0 = std::make_unique<core::CpuField>(domain.sub, "dP0");      // zero-init
    poi.engine->solve_nob(dt,
                          fields[core::Var::U], fields[core::Var::V],
                          fields[core::Var::W], fields[core::Var::P],
                          inv_rho, *poi.dPhat, *poi.dP0);
  } else {
    poi.engine->solve(dt, fields[core::Var::U], fields[core::Var::V], fields[core::Var::W], fields[core::Var::P]);
  }
}

} // namespace mpmstd::equation
