#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// Solve the assembled scalar ADI system: 3-sweep ADI (sweep order from BC,
// distributed (P)TDMA via Domain::tdma, wall rows via modify_tdma_row_cpu) then
// add the increment back into Fields[Var::T]. Caller syncs T afterward. (rev.2 §5)
void solve_scalar_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                      core::CpuFields& fields, core::CpuScalarSystem& scalar_system, real_t dt);
void solve_scalar_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                      core::GpuFields& fields, core::GpuScalarSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation
