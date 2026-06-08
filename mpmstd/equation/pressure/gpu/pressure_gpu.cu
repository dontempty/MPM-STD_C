#include "equation/pressure/assemble.hpp"
#include "equation/pressure/solve.hpp"

// P0 skeleton stubs (no-op), GPU build only. P5 ports the device kernels.

namespace mpmstd::equation {

void assemble_pressure_system_gpu(core::PressureSystem&, const core::GpuField&, const core::GpuField&,
                                  const core::GpuField&, const core::Grid&, real_t) { /* TODO(P5) */ }

void solve_pressure_gpu(core::PressureSystem&, core::GpuField&, const core::Subdomain&) { /* TODO(P5) */ }

void project_velocity_gpu(core::GpuField&, core::GpuField&, core::GpuField&, core::GpuField&,
                          const core::GpuField&, const core::Grid&, real_t) { /* TODO(P5) */ }

} // namespace mpmstd::equation
