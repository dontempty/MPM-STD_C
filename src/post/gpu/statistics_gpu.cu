#include "post/statistics.hpp"

// P1/P5 stub (no-op), GPU build only. Device plane-reduction at P5.
namespace mpmstd::post {
void accumulate_statistics_gpu(Stats&, const core::Domain&, const core::GpuFields&) { /* TODO(P5) */ }
} // namespace mpmstd::post
