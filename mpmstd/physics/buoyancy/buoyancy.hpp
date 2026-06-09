#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"

namespace mpmstd::physics {

// OB/NOB Boussinesq buoyancy. Composed in main AFTER momentum assemble (adds to
// the RHS b). Omit the call ⇒ no buoyancy. (rev.2 §9)
struct BuoyancyParams {
  int    gravity_axis = 2;   // wall/gravity axis is case-dependent (channel z, DHVC streamwise, RBC vertical)
  real_t coeff        = 0;   // Ra/Pr-derived scaling (set per case)
  real_t T_ref        = 0;
};

void add_buoyancy_force_cpu(core::CpuMomentumSystem& mom, const core::CpuField& T, const BuoyancyParams& p, real_t dt);
void add_buoyancy_force_gpu(core::GpuMomentumSystem& mom, const core::GpuField& T, const BuoyancyParams& p, real_t dt);

} // namespace mpmstd::physics
