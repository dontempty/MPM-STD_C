#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"

namespace mpmstd::equation {

// Momentum assemble — momentum is T-INDEPENDENT (rev.2 §9). Builds the explicit
// MPM-STD BW-ADI RHS per component into the MomentumSystem (compute_mpmstd_rhs):
//   rhs[c] = dt·nu·(Lx+Ly+Lz)q^n − dt·div(u·q^n).
// Pressure gradient is applied by the projection (+ channel forcing), NOT here.
// Two variants:
//   const_visc : scalar nu, NO cross-stress  → channel / cavity / DHVC (P1)
//   var_visc   : mu field,  WITH cross-stress → NOB / LES (P7)
void assemble_momentum_const_visc_cpu(core::MomentumSystem& mom,
                                      const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                      const core::Grid& grid, real_t nu, real_t dt);
void assemble_momentum_var_visc_cpu  (core::MomentumSystem& mom,
                                      const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                      const core::CpuField& mu, const core::Grid& grid, real_t dt);

void assemble_momentum_const_visc_gpu(core::MomentumSystem& mom,
                                      const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                      const core::Grid& grid, real_t nu, real_t dt);
void assemble_momentum_var_visc_gpu  (core::MomentumSystem& mom,
                                      const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                      const core::GpuField& mu, const core::Grid& grid, real_t dt);

} // namespace mpmstd::equation
