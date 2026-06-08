#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::physics {

// Channel forcing (rev.2 U7): constant dP/dx body force into the momentum RHS,
// + bulk mass-flow correction to hold Ub. Composed in main; omit ⇒ no forcing.
void apply_pressure_gradient_cpu(core::MomentumSystem& mom, real_t dpdx, real_t dt);
void apply_pressure_gradient_gpu(core::MomentumSystem& mom, real_t dpdx, real_t dt);

void apply_mass_flow_correction_cpu(core::CpuField& U, real_t Ub_target, const core::Subdomain& sub, real_t dt);
void apply_mass_flow_correction_gpu(core::GpuField& U, real_t Ub_target, const core::Subdomain& sub, real_t dt);

} // namespace mpmstd::physics
