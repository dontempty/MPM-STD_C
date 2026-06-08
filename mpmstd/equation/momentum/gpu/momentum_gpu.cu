#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"

// P0 skeleton stubs (no-op), GPU build only. P5 ports the device kernels.

namespace mpmstd::equation {

void assemble_momentum_const_visc_gpu(core::MomentumSystem&, const core::GpuField&, const core::GpuField&,
                                      const core::GpuField&, const core::GpuField&, real_t,
                                      const core::Grid&, const core::Boundary&, real_t) { /* TODO(P5) */ }

void assemble_momentum_var_visc_gpu(core::MomentumSystem&, const core::GpuField&, const core::GpuField&,
                                    const core::GpuField&, const core::GpuField&, const core::GpuField&,
                                    const core::Grid&, const core::Boundary&, real_t) { /* TODO(P5/P7) */ }

void solve_momentum_gpu(core::MomentumSystem&, const core::GpuField&, const core::GpuField&, const core::GpuField&,
                        core::GpuField&, core::GpuField&, core::GpuField&, const core::Subdomain&) { /* TODO(P5) */ }

void update_velocity_gpu(core::GpuField&, core::GpuField&, core::GpuField&,
                         const core::GpuField&, const core::GpuField&, const core::GpuField&) { /* TODO(P5) */ }

} // namespace mpmstd::equation
