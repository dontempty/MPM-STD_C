#include "equation/pressure/solve.hpp"

// P1/P4 skeleton stub (no-op), GPU build only. P4 ports the cuFFT + device-TDMA
// pencil pipeline (signature tracks the CPU side: Domain + GpuFields).

namespace mpmstd::equation {

void solve_pressure_gpu(const core::Domain&, const core::BoundaryCondition&,
                        core::GpuFields&, core::PressureSystem&, real_t) { /* TODO(P4) */ }

} // namespace mpmstd::equation
