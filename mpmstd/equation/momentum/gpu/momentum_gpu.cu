#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"

// P1/P5 skeleton stubs (no-op), GPU build only. Signatures track the CPU side
// (Domain + GpuFields + GpuMomentumSystem). Real device kernels land in P5.

namespace mpmstd::equation {

void assemble_momentum_const_visc_gpu(const core::Domain&, core::GpuFields&, core::GpuMomentumSystem&, real_t) {}
void assemble_momentum_var_visc_gpu  (const core::Domain&, core::GpuFields&, core::GpuMomentumSystem&, real_t) {}
void solve_momentum_gpu(const core::Domain&, const core::BoundaryCondition&, core::GpuFields&, core::GpuMomentumSystem&, real_t) {}
void update_velocity_gpu(core::GpuFields&, const core::GpuMomentumSystem&) {}

} // namespace mpmstd::equation
