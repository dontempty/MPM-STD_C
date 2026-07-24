#pragma once

#include "core/domain.hpp"
#include "core/fields.hpp"
#include "equation/scalar/linear_system.hpp"

namespace mpmstd::equation::scalar {

// OB(CPU) coefficient step — builds the explicit Crank-Nicolson RHS (full
// Laplacian + conservative convection) into the ScalarSystem before
// solve_scalar_constant_property_cpu. Reads T,U,V,W from Fields. (rev.2 §5)
// NOTE: this legacy OB-CPU scheme is the ONLY remaining separate coef step.
// The nob (faithful) path and the GPU OB wrapper build the RHS fused with the
// A bands inside their solve (build_A_rhs_nob / k_build_A_rhs_nob) — no coef
// call needed there.
void coef_scalar_constant_property_cpu(const core::Domain& domain, core::CpuFields& fields,
                        equation::scalar::CpuLinearSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation::scalar
