#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// rev.2 M2: solve_momentum does U,V,W + the block lower-triangular velocity
// coupling in one (MPM-STD fortran core_momentum). Increments are written into
// the MomentumSystem's own dU,dV,dW (created inside momentum). Distributed
// (P)TDMA via Domain::tdma; BC via core::modify_tdma_row_cpu + sync.
void solve_momentum_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::CpuFields& fields, core::CpuMomentumSystem& mom, real_t dt);
void solve_momentum_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::GpuFields& fields, core::GpuMomentumSystem& mom, real_t dt);

// pseudo-update: U += dU (etc.); caller syncs U,V,W afterwards.
void update_velocity_cpu(core::CpuFields& fields, const core::CpuMomentumSystem& mom);
void update_velocity_gpu(core::GpuFields& fields, const core::GpuMomentumSystem& mom);

} // namespace mpmstd::equation
