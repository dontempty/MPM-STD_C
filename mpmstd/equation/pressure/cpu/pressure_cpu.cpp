#include "equation/pressure/solve.hpp"
#include "equation/pressure/pressure_engine.hpp"   // PressureSolver (complete type for make_shared)

#include <memory>

// P1 — CPU pressure step. Lazily builds the validated pencil-FFT PressureSolver
// engine (held by PressureSystem) and runs the full step: div(U*) → C→I→FFTx→
// I→Y→FFTy → distributed z-TDMA → IFFT → I→C → unpack dP into P → project U,V,W.

namespace mpmstd::equation {

void solve_pressure_cpu(core::PressureSystem& poi, real_t dt,
                        core::CpuField& U, core::CpuField& V, core::CpuField& W, core::CpuField& P,
                        const core::Grid& grid, const core::Boundary& problem,
                        linear_solver::tdma::TdmaRegistry& tdma, const core::Subdomain& sub) {
  if (!poi.engine)
    poi.engine = std::make_shared<pressure::PressureSolver>(grid, sub, problem, tdma);
  poi.engine->solve(dt, U, V, W, P);
}

} // namespace mpmstd::equation
