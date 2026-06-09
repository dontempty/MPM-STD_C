#pragma once

#include "core/domain.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// Momentum assemble — T-independent (rev.2 §9). Reads the velocity fields + the
// nu constant from `Fields`, builds the explicit MPM-STD BW-ADI RHS per component
// into the MomentumSystem (compute_mpmstd_rhs). Pressure gradient is applied by
// projection (+ channel forcing), NOT here.
//   const_visc : scalar nu (Fields constant), NO cross-stress  → channel/cavity/DHVC (P1)
//   var_visc   : mu field,  WITH cross-stress → NOB/LES (P7)
void assemble_momentum_const_visc_cpu(const core::Domain& domain, core::CpuFields& fields,
                                      core::CpuMomentumSystem& mom, real_t dt);
void assemble_momentum_var_visc_cpu  (const core::Domain& domain, core::CpuFields& fields,
                                      core::CpuMomentumSystem& mom, real_t dt);

void assemble_momentum_const_visc_gpu(const core::Domain& domain, core::GpuFields& fields,
                                      core::GpuMomentumSystem& mom, real_t dt);
void assemble_momentum_var_visc_gpu  (const core::Domain& domain, core::GpuFields& fields,
                                      core::GpuMomentumSystem& mom, real_t dt);

} // namespace mpmstd::equation
