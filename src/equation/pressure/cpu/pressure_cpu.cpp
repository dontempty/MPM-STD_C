#include "equation/pressure/solve.hpp"
#include "equation/pressure/pressure_engine.hpp"   // PressureSolver (complete type)

#include <memory>

// P1 pressure step rewired to Domain + Fields. Lazily builds the validated
// pencil-FFT engine (from Domain+BC) and runs the full step on the Fields.

namespace mpmstd::equation {

void solve_pressure_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::CpuFields& fields, core::PressureSystem& poi, real_t dt,
                        const real_t* inv_rho) {
  if (!poi.engine)
    poi.engine = std::make_shared<pressure::PressureSolver>(domain.grid, domain.sub, bc, domain.tdma);
  if (inv_rho) {
    if (!poi.dPhat)
      poi.dPhat = std::make_unique<core::CpuField>(domain.sub, "dPhat");
    poi.engine->solve_nob(dt,
                          fields[core::Var::U], fields[core::Var::V],
                          fields[core::Var::W], fields[core::Var::P],
                          inv_rho, *poi.dPhat);
  } else {
    poi.engine->solve(dt, fields[core::Var::U], fields[core::Var::V], fields[core::Var::W], fields[core::Var::P]);
  }
}

} // namespace mpmstd::equation
