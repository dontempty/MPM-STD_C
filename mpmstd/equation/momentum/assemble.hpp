#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/boundary.hpp"

namespace mpmstd::equation {

// Momentum assemble — momentum is T-INDEPENDENT (rev.2 §9). Two variants:
//   const_visc : scalar nu, NO cross-stress  → channel / cavity / DHVC (optimal)
//   var_visc   : mu field,  WITH cross-stress → NOB / LES (the deferred terms)
// The caller picks the right one. (bodies: P1 const, P7 var)
void assemble_momentum_const_visc_cpu(core::MomentumSystem& mom,
                                      const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                      const core::CpuField& P, real_t nu,
                                      const core::Grid& grid, const core::Boundary& bc, real_t dt);
void assemble_momentum_var_visc_cpu  (core::MomentumSystem& mom,
                                      const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                      const core::CpuField& P, const core::CpuField& mu,
                                      const core::Grid& grid, const core::Boundary& bc, real_t dt);

void assemble_momentum_const_visc_gpu(core::MomentumSystem& mom,
                                      const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                      const core::GpuField& P, real_t nu,
                                      const core::Grid& grid, const core::Boundary& bc, real_t dt);
void assemble_momentum_var_visc_gpu  (core::MomentumSystem& mom,
                                      const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                                      const core::GpuField& P, const core::GpuField& mu,
                                      const core::Grid& grid, const core::Boundary& bc, real_t dt);

} // namespace mpmstd::equation
