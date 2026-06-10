#include "physics/properties/properties.hpp"

// P5 stub (no-op), GPU build only. Device property kernel lands at P5.
namespace mpmstd::physics {
void update_properties_gpu(GpuPropertyFields&, const core::GpuField&, const PropertyModel&) { /* TODO(P5) */ }
} // namespace mpmstd::physics
