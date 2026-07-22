#pragma once

#include "core/domain.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// OB(CPU) coefficient step — builds the explicit Crank-Nicolson RHS (full
// Laplacian + conservative convection) into the ScalarSystem before
// solve_scalar_ob_cpu. Reads T,U,V,W from Fields. (rev.2 §5)
// NOTE: this legacy OB-CPU scheme is the ONLY remaining separate coef step.
// The nob (faithful) path and the GPU OB wrapper build the RHS fused with the
// A bands inside their solve (build_A_rhs_nob / k_build_A_rhs_nob) — no coef
// call needed there.
void coef_scalar_ob_cpu(const core::Domain& domain, core::CpuFields& fields,
                        core::CpuScalarSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation
