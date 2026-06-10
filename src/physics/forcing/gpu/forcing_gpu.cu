#include "physics/forcing/forcing.hpp"

// P1/P5 skeleton stubs (no-op), GPU build only. Device kernels + reductions at P5.

namespace mpmstd::physics {

void apply_body_force_gpu(core::GpuFields&, real_t, real_t) { /* TODO(P5) */ }

double channel_bulk_velocity_gpu(const core::Domain&, const core::GpuFields&, double) { return 0.0; /* TODO(P5) */ }

double apply_mass_flow_correction_gpu(const core::Domain&, core::GpuFields&,
                                      double, double, double, double& dpdx) { return dpdx; /* TODO(P5) */ }

} // namespace mpmstd::physics
