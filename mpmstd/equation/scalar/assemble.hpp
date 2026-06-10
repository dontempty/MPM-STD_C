#pragma once

#include "core/domain.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// Scalar (energy) transport assemble — builds the explicit Crank-Nicolson RHS
// (full Laplacian + conservative convection) into the ScalarSystem. Reads
// T,U,V,W from Fields. (rev.2 §5)
//   const_diff : scalar diffusivity alpha (Fields constant) → OB DHVC/channel (P6)
//   var_diff   : kappa field (NOB variable conductivity) → RBC glycerol (P7)
void assemble_scalar_const_diff_cpu(const core::Domain& domain, core::CpuFields& fields,
                                    core::CpuScalarSystem& scalar_system, real_t dt);
void assemble_scalar_const_diff_gpu(const core::Domain& domain, core::GpuFields& fields,
                                    core::GpuScalarSystem& scalar_system, real_t dt);

// Variable-conductivity (NOB) twin: same explicit RHS but with face-κ̂ diffusion
// and the 1/(ρ̂Ĉp) prefactor (laplacian_explicit_rhs_var); convection unchanged.
// With κ̂≡1, invρĈp≡1 it reproduces assemble_scalar_const_diff_cpu. Ct=alpha_T.
void assemble_scalar_var_diff_cpu(const core::Domain& domain, core::CpuFields& fields,
                                  const core::CpuField& kappa, const core::CpuField& inv_rhocp,
                                  core::CpuScalarSystem& scalar_system, real_t dt);
void assemble_scalar_var_diff_gpu(const core::Domain& domain, core::GpuFields& fields,
                                  const core::GpuField& kappa, const core::GpuField& inv_rhocp,
                                  core::GpuScalarSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation
