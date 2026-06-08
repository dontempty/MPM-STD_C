#include "post/diagnostics.hpp"

// P0 skeleton stubs (no-op), GPU build only.
namespace mpmstd::post {
void compute_Re_delta_star_gpu(real_t& out, const core::GpuField&, const core::Grid&, const core::Subdomain&) { out = 0; /* TODO(P6) */ }
void compute_center_temp_gpu  (real_t& out, const core::GpuField&, const core::Grid&, const core::Subdomain&) { out = 0; /* TODO(P7) */ }
} // namespace mpmstd::post
