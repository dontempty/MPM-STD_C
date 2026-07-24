#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/variables.hpp"
#include "equation/scalar/linear_system.hpp"

namespace mpmstd::equation::scalar {

// Solve the assembled scalar ADI system: 3-sweep ADI (sweep order from BC,
// distributed (P)TDMA via Domain::tdma, wall rows via modify_tdma_row_cpu) then
// add the increment back into Fields[Var::T]. Caller syncs T afterward. (rev.2 §5)
void solve_scalar_ob_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                         core::CpuFields& fields, equation::scalar::CpuLinearSystem& scalar_system, real_t dt);
void solve_scalar_ob_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                         core::GpuFields& fields, equation::scalar::GpuLinearSystem& scalar_system, real_t dt);

// Variable-κ̂ (NOB) twin — FAITHFUL PaScaL_TCS delta-form 3-factor ADI, fused
// build_A_RHS: each sweep builds bands AND packs the RHS in one pass
// (build_A_rhs_nob / k_build_A_rhs_nob) carrying that axis's diffusion + dκ̂
// linearization + 0.25 conservative-face semi-implicit convection; stage 1
// computes the explicit residual R inline and packs the M·θⁿ delta-form RHS
// (no separate coef step). Needs dKPP + dinvRhoCp. With props≡1 the
// diffusion reduces to solve_scalar_ob_cpu's. Ct=alpha_T (=1/√(Ra·Pr)).
void solve_scalar_nob_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                          core::CpuFields& fields,
                          const core::CpuField& KPP, const core::CpuField& dKPP,
                          const core::CpuField& invRhoCp, const core::CpuField& dinvRhoCp,
                          equation::scalar::CpuLinearSystem& scalar_system, real_t dt);
void solve_scalar_nob_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                          core::GpuFields& fields,
                          const core::GpuField& KPP, const core::GpuField& dKPP,
                          const core::GpuField& invRhoCp, const core::GpuField& dinvRhoCp,
                          equation::scalar::GpuLinearSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation::scalar
