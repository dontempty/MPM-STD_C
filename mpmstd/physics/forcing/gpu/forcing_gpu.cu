#include "physics/forcing/forcing.hpp"

// P1 skeleton stubs (no-op), GPU build only. P5 ports the device kernels +
// reductions (apply_body_force / bulk velocity / mass-flow correction).

namespace mpmstd::physics {

void apply_body_force_gpu(core::GpuField&, real_t, real_t) { /* TODO(P5) */ }

double channel_bulk_velocity_gpu(const core::GpuField&, const core::Grid&,
                                 const core::Subdomain&, double) { return 0.0; /* TODO(P5) */ }

double apply_mass_flow_correction_gpu(core::GpuField&, double, const core::Grid&,
                                      const core::Subdomain&, double, double, double& dpdx) {
  return dpdx; /* TODO(P5) */
}

} // namespace mpmstd::physics
