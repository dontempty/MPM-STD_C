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

// Variable-κ̂ (NOB) twin — FAITHFUL PaScaL_TCS delta-form 3-factor ADI: each
// sweep builds bands (build_energy_adi_bands_var) carrying that axis's diffusion
// + dκ̂ linearization + 0.25 conservative-face semi-implicit convection; stage 1
// packs the M·θⁿ delta-form RHS. Needs dkappa + dinv_rhocp. With props≡1 the
// diffusion reduces to solve_scalar_cpu's. Ct=alpha_T (=1/√(Ra·Pr)).
void solve_scalar_var_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                          core::CpuFields& fields,
                          const core::CpuField& kappa, const core::CpuField& dkappa,
                          const core::CpuField& inv_rhocp, const core::CpuField& dinv_rhocp,
                          core::CpuScalarSystem& scalar_system, real_t dt);
void solve_scalar_var_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                          core::GpuFields& fields,
                          const core::GpuField& kappa, const core::GpuField& dkappa,
                          const core::GpuField& inv_rhocp, const core::GpuField& dinv_rhocp,
                          core::GpuScalarSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation
