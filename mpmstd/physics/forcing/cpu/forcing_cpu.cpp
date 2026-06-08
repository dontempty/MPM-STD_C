#include "physics/forcing/forcing.hpp"

// P0 skeleton stubs (no-op). P1/P2 fill dP/dx forcing + mass-flow correction.
namespace mpmstd::physics {
void apply_pressure_gradient_cpu(core::MomentumSystem&, real_t, real_t) { /* TODO(P1) */ }
void apply_mass_flow_correction_cpu(core::CpuField&, real_t, const core::Subdomain&, real_t) { /* TODO(P1) */ }
} // namespace mpmstd::physics
