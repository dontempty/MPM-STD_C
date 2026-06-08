#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/boundary.hpp"                        // Boundary = boundary::Problem
#include "core/mpi_topology.hpp"                    // Subdomain
#include "linear_solver/tdma/tdma_registry.hpp"     // reused distributed (P)TDMA

namespace mpmstd::equation {

// rev.2 M2: solve_momentum does U,V,W + dU,dV,dW + the block lower-triangular
// velocity coupling ALL IN ONE — matching the MPM-STD fortran core_momentum
// (solvedU/V/W + blockLdV/blockLdU). No separate couple call in main.
//
// Reuses the distributed TdmaRegistry (PaScaL). BC via the ported free funcs
// core::modify_tdma_row_cpu (band fold) + core::apply_ghost_cpu (increment
// ghost for the coupling stencil). On return dU,dV,dW hold the block-coupled
// increments with halos+ghosts filled; update_velocity_cpu applies them.
void solve_momentum_cpu(core::MomentumSystem& mom,
                        const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                        core::CpuField& dU, core::CpuField& dV, core::CpuField& dW,
                        const core::Grid& grid, const core::Boundary& problem,
                        linear_solver::tdma::TdmaRegistry& tdma, const core::Subdomain& sub,
                        real_t nu, real_t dt);

// pseudo-update q += dq (interior only; caller does halo + BC ghost on U,V,W).
void update_velocity_cpu(core::CpuField& U, core::CpuField& V, core::CpuField& W,
                         const core::CpuField& dU, const core::CpuField& dV, const core::CpuField& dW);

// GPU placeholders (P5).
void solve_momentum_gpu(core::MomentumSystem& mom,
                        const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                        core::GpuField& dU, core::GpuField& dV, core::GpuField& dW,
                        const core::Subdomain& sub);
void update_velocity_gpu(core::GpuField& U, core::GpuField& V, core::GpuField& W,
                         const core::GpuField& dU, const core::GpuField& dV, const core::GpuField& dW);

} // namespace mpmstd::equation
