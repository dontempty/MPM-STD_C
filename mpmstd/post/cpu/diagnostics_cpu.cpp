#include "post/diagnostics.hpp"

// P0 skeleton stubs (no-op). P6 fills Re_δ* (Fig 7); P7 fills Θ_c (Fig 9).
namespace mpmstd::post {
void compute_Re_delta_star_cpu(real_t& out, const core::CpuField&, const core::Grid&, const core::Subdomain&) { out = 0; /* TODO(P6) */ }
void compute_center_temp_cpu  (real_t& out, const core::CpuField&, const core::Grid&, const core::Subdomain&) { out = 0; /* TODO(P7) */ }
} // namespace mpmstd::post
