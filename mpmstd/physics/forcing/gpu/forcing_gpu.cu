#include "physics/forcing/forcing.hpp"

// P0 skeleton stubs (no-op), GPU build only.
namespace mpmstd::physics {
void apply_pressure_gradient_gpu(core::MomentumSystem&, real_t, real_t) { /* TODO(P5) */ }
void apply_mass_flow_correction_gpu(core::GpuField&, real_t, const core::Subdomain&, real_t) { /* TODO(P5) */ }
} // namespace mpmstd::physics
