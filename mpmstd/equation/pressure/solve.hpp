#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::equation {

// Solve pressure-Poisson (calls solve/poisson_solver: FFT/DCT/TDMA chosen from
// BC per axis, rev.2 §9c), then project velocity divergence-free + update P
// (projection ALWAYS runs, rev.2 §7/U7).
void solve_pressure_cpu  (core::PressureSystem& poi, core::CpuField& dP, const core::Subdomain& sub);
void solve_pressure_gpu  (core::PressureSystem& poi, core::GpuField& dP, const core::Subdomain& sub);

void project_velocity_cpu(core::CpuField& U, core::CpuField& V, core::CpuField& W, core::CpuField& P,
                          const core::CpuField& dP, const core::Grid& grid, real_t dt);
void project_velocity_gpu(core::GpuField& U, core::GpuField& V, core::GpuField& W, core::GpuField& P,
                          const core::GpuField& dP, const core::Grid& grid, real_t dt);

} // namespace mpmstd::equation
