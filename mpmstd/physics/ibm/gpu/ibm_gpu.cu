#include "physics/ibm/ibm.hpp"

// P0 skeleton stubs (no-op), GPU build only.
namespace mpmstd::physics {
void build_ibm_mask_gpu(IbmMask&, const core::Grid&) { /* TODO(P10) */ }
void apply_ibm_forcing_gpu(core::MomentumSystem&, const IbmMask&) { /* TODO(P10) */ }
} // namespace mpmstd::physics
