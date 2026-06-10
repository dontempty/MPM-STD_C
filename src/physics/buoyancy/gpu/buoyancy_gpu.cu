#include "physics/buoyancy/buoyancy.hpp"

// P0 skeleton stub (no-op), GPU build only.
namespace mpmstd::physics {
void add_buoyancy_force_gpu(core::GpuMomentumSystem&, const core::GpuField&, const BuoyancyParams&, real_t) { /* TODO(P6) */ }
void add_buoyancy_force_nob_gpu(core::GpuMomentumSystem&, const core::GpuField&, const core::GpuField&, const BuoyancyParams&, real_t) { /* TODO(P6) */ }
} // namespace mpmstd::physics
