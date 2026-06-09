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

} // namespace mpmstd::equation
