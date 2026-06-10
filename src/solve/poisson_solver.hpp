#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::solve {

// Distributed pressure-Poisson solve (common, implicit). Per-axis transform is
// chosen from the BC (rev.2 §9c): periodic→FFT, Neumann/wall+uniform→DCT, the
// remaining axis→TDMA; cavity (all-Neumann) adds null-space pinning. Transpose
// + transforms are device-to-device under CUDA-aware MPI in the gpu variant.
// (bodies: P3 cpu, P4 gpu)
void solve_poisson_cpu(core::PressureSystem& poi, core::CpuField& dP, const core::Subdomain& sub);
void solve_poisson_gpu(core::PressureSystem& poi, core::GpuField& dP, const core::Subdomain& sub);

} // namespace mpmstd::solve
