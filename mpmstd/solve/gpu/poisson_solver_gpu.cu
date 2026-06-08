#include "solve/poisson_solver.hpp"

// P0 skeleton stub (no-op), GPU build only. P4 ports the cuFFT + device-TDMA
// pipeline with CUDA-aware MPI transposes.
namespace mpmstd::solve {
void solve_poisson_gpu(core::PressureSystem&, core::GpuField&, const core::Subdomain&) { /* TODO(P4) */ }
} // namespace mpmstd::solve
