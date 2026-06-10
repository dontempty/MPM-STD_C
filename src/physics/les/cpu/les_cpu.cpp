#include "physics/les/les.hpp"

// P0 skeleton stub (no-op). P9 fills the SGS model (e.g. Smagorinsky/WALE).
namespace mpmstd::physics {
void compute_sgs_viscosity_cpu(core::CpuField&, const core::CpuField&, const core::CpuField&,
                               const core::CpuField&, const core::Grid&) { /* TODO(P9) */ }
} // namespace mpmstd::physics
