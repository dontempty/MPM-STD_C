#include "equation/pressure/solve.hpp"

// P0/P1 skeleton stub (no-op), GPU build only. P4 ports the cuFFT + device-TDMA
// pencil pipeline with CUDA-aware MPI transposes (signature tracks the CPU side).

namespace mpmstd::equation {

void solve_pressure_gpu(core::PressureSystem&, real_t,
                        core::GpuField&, core::GpuField&, core::GpuField&, core::GpuField&,
                        const core::Grid&, const core::Boundary&,
                        linear_solver::tdma::TdmaRegistry&, const core::Subdomain&) { /* TODO(P4) */ }

} // namespace mpmstd::equation
