#include "physics/ibm/ibm.hpp"

// P0 skeleton stubs (no-op). P10 fills mask build + IB forcing.
namespace mpmstd::physics {
void build_ibm_mask_cpu(IbmMask&, const core::Grid&) { /* TODO(P10) */ }
void apply_ibm_forcing_cpu(core::CpuMomentumSystem&, const IbmMask&) { /* TODO(P10) */ }
} // namespace mpmstd::physics
