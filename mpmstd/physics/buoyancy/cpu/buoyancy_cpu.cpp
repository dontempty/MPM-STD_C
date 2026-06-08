#include "physics/buoyancy/buoyancy.hpp"

// P0 skeleton stub (no-op). P6 (OB) / P7 (NOB) fill the buoyancy source.
namespace mpmstd::physics {
void add_buoyancy_force_cpu(core::MomentumSystem&, const core::CpuField&, const BuoyancyParams&, real_t) { /* TODO(P6) */ }
} // namespace mpmstd::physics
