#include "post/statistics.hpp"

// P0 skeleton stub (no-op). P1 ports the plane-averaged profiles (GLOBAL nx*ny
// normalization — guard the 16x bug).
namespace mpmstd::post {
void accumulate_statistics_cpu(Stats&, const core::CpuField&, const core::CpuField&,
                               const core::CpuField&, const core::CpuField&, const core::Subdomain&) { /* TODO(P1) */ }
} // namespace mpmstd::post
