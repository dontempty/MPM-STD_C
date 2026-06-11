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

// Variable-κ̂ (NOB) twin — FAITHFUL PaScaL_TCS port: builds only the explicit
// residual R (energy_residual_rhs_var; convection is now 0.5-skew advective and
// part of R). The semi-implicit convection + delta-form pack live in the solve.
// Needs dkappa + dinv_rhocp for the dκ̂/dinvρĈp linearizations. dt is applied in
// the band pack, NOT here. With props≡1 the diffusion reduces to the const path.
void assemble_scalar_var_diff_cpu(const core::Domain& domain, core::CpuFields& fields,
                                  const core::CpuField& kappa, const core::CpuField& dkappa,
                                  const core::CpuField& inv_rhocp, const core::CpuField& dinv_rhocp,
                                  core::CpuScalarSystem& scalar_system, real_t dt);
void assemble_scalar_var_diff_gpu(const core::Domain& domain, core::GpuFields& fields,
                                  const core::GpuField& kappa, const core::GpuField& dkappa,
                                  const core::GpuField& inv_rhocp, const core::GpuField& dinv_rhocp,
                                  core::GpuScalarSystem& scalar_system, real_t dt);

} // namespace mpmstd::equation
