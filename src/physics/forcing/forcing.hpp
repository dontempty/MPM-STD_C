#pragma once

#include "core/domain.hpp"
#include "core/variables.hpp"

namespace mpmstd::physics {

// Channel forcing (rev.2 U7). The mean dP/dx STATE lives in the caller; these
// free functions read/update it explicitly. Operate on Fields[U] + the Domain.
void apply_body_force_cpu(core::CpuFields& fields, real_t dpdx, real_t dt);   // U += -dt*dpdx
void apply_body_force_gpu(core::GpuFields& fields, real_t dpdx, real_t dt);

double channel_total_volume_cpu(const core::Domain& domain);                  // interior volume (once)

double channel_bulk_velocity_cpu(const core::Domain& domain, const core::CpuFields& fields, double total_vol);
double channel_bulk_velocity_gpu(const core::Domain& domain, const core::GpuFields& fields, double total_vol);

// MassFlow: shift U so bulk==Ub_target; dpdx += (Ub-Ub_target)/dt. Returns dpdx.
double apply_mass_flow_correction_cpu(const core::Domain& domain, core::CpuFields& fields,
                                      double Ub_target, double dt, double total_vol, double& dpdx);
double apply_mass_flow_correction_gpu(const core::Domain& domain, core::GpuFields& fields,
                                      double Ub_target, double dt, double total_vol, double& dpdx);

} // namespace mpmstd::physics
