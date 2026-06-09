#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::physics {

// Channel forcing (rev.2 U7). The mean dP/dx STATE lives in the caller (main),
// not in a class — these free functions read/update it explicitly. Composed in
// main; omit ⇒ no forcing.
//
//   PressureGradient mode : dpdx constant (= -|target|), apply_body_force each step.
//   MassFlow mode         : apply_body_force then apply_mass_flow_correction, which
//                           shifts U to hold Ub_target and evolves dpdx.

// U_interior += -dt*dpdx  (dpdx<0 drives +x).
void apply_body_force_cpu(core::CpuField& U, real_t dpdx, real_t dt);
void apply_body_force_gpu(core::GpuField& U, real_t dpdx, real_t dt);

// Total interior volume (compute once; for bulk averaging).
double channel_total_volume_cpu(const core::Grid& grid, const core::Subdomain& sub);

// Volume-averaged bulk streamwise velocity (cell-center U = 0.5*(U[i]+U[i+1])).
double channel_bulk_velocity_cpu(const core::CpuField& U, const core::Grid& grid,
                                 const core::Subdomain& sub, double total_vol);
double channel_bulk_velocity_gpu(const core::GpuField& U, const core::Grid& grid,
                                 const core::Subdomain& sub, double total_vol);

// MassFlow correction: shift U so bulk==Ub_target; dpdx += (Ub-Ub_target)/dt.
// dpdx updated in place; returns the new dpdx.
double apply_mass_flow_correction_cpu(core::CpuField& U, double Ub_target,
                                      const core::Grid& grid, const core::Subdomain& sub,
                                      double dt, double total_vol, double& dpdx);
double apply_mass_flow_correction_gpu(core::GpuField& U, double Ub_target,
                                      const core::Grid& grid, const core::Subdomain& sub,
                                      double dt, double total_vol, double& dpdx);

} // namespace mpmstd::physics
