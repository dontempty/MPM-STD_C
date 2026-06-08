#include "equation/pressure/assemble.hpp"
#include "equation/pressure/solve.hpp"

// P0 skeleton stubs (no-op). P1/P3 port RHS+projection; the transform machinery
// lives in solve/poisson_solver.

namespace mpmstd::equation {

void assemble_pressure_system_cpu(core::PressureSystem&, const core::CpuField&, const core::CpuField&,
                                  const core::CpuField&, const core::Grid&, real_t) { /* TODO(P1) */ }

void solve_pressure_cpu(core::PressureSystem&, core::CpuField&, const core::Subdomain&) { /* TODO(P1) */ }

void project_velocity_cpu(core::CpuField&, core::CpuField&, core::CpuField&, core::CpuField&,
                          const core::CpuField&, const core::Grid&, real_t) { /* TODO(P1) */ }

} // namespace mpmstd::equation
