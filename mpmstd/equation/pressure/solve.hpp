#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/boundary.hpp"                      // Boundary = boundary::Problem
#include "core/mpi_topology.hpp"                  // Subdomain
#include "linear_solver/tdma/tdma_registry.hpp"

namespace mpmstd::equation {

// Pressure step (rev.2 §7). For P1 the divergence RHS + FFT/transpose/TDMA
// Poisson + dP unpack (into P) + velocity projection run as ONE call — matching
// the validated PressureSolver exactly. The heavy engine (FFTW plans, pencil
// buffers, wavenumbers, distributed z-TDMA) is held by PressureSystem, built
// lazily on first call and reused. [P2 may re-split assemble/solve/project for
// the §7 recipe once the bundled version regresses.]
void solve_pressure_cpu(core::PressureSystem& poi, real_t dt,
                        core::CpuField& U, core::CpuField& V, core::CpuField& W, core::CpuField& P,
                        const core::Grid& grid, const core::Boundary& problem,
                        linear_solver::tdma::TdmaRegistry& tdma, const core::Subdomain& sub);

void solve_pressure_gpu(core::PressureSystem& poi, real_t dt,
                        core::GpuField& U, core::GpuField& V, core::GpuField& W, core::GpuField& P,
                        const core::Grid& grid, const core::Boundary& problem,
                        linear_solver::tdma::TdmaRegistry& tdma, const core::Subdomain& sub);

} // namespace mpmstd::equation
