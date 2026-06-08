#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::equation {

// rev.2 M2: solve_momentum does U,V,W + dU,dV,dW + the block lower-triangular
// velocity coupling ALL IN ONE — matching the MPM-STD fortran core_momentum
// (solvedU/V/W + blockLdV/blockLdU). No separate couple call in main.
void solve_momentum_cpu(core::MomentumSystem& mom,
                        const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                        core::CpuField& dU, core::CpuField& dV, core::CpuField& dW,
                        const core::Subdomain& sub);
void solve_momentum_gpu(core::MomentumSystem& mom,
                        const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                        core::GpuField& dU, core::GpuField& dV, core::GpuField& dW,
                        const core::Subdomain& sub);

// pseudo-update U += dU (etc.)
void update_velocity_cpu(core::CpuField& U, core::CpuField& V, core::CpuField& W,
                         const core::CpuField& dU, const core::CpuField& dV, const core::CpuField& dW);
void update_velocity_gpu(core::GpuField& U, core::GpuField& V, core::GpuField& W,
                         const core::GpuField& dU, const core::GpuField& dV, const core::GpuField& dW);

} // namespace mpmstd::equation
