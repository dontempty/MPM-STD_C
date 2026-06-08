#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"

namespace mpmstd::physics {

// NOB temperature-dependent properties μ(T), 1/ρ(T), κ(T), 1/(ρCp)(T). Composed
// in main BEFORE assemble (produces the effective-coefficient fields). Omit ⇒
// constant properties (OB). Polynomial coeffs (Pan et al. for glycerol). (§9)
struct PropertyModel {
  // polynomial coefficients filled in P7; placeholder for the skeleton
  real_t mu0 = 1, rho0 = 1, cp0 = 1, kappa0 = 1;
};

void update_properties_cpu(core::CpuField& mu, core::CpuField& irho, core::CpuField& kappa,
                           core::CpuField& irhocp, const core::CpuField& T, const PropertyModel& m);
void update_properties_gpu(core::GpuField& mu, core::GpuField& irho, core::GpuField& kappa,
                           core::GpuField& irhocp, const core::GpuField& T, const PropertyModel& m);

} // namespace mpmstd::physics
