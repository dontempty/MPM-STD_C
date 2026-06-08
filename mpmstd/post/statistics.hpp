#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::post {

// Running turbulence statistics (plane-averaged profiles + rms). (rev.2 §5)
// ⚠ Normalization MUST use the GLOBAL nx*ny when MPI_SUM-reducing over the
// homogeneous-plane ranks — dividing by per-rank n_interior caused the 16x bug.
// (see memory: channel-wss-diagnostic / channel_stats.hpp fix)
struct Stats {
  long  n_samples = 0;
  // z-profiles of <U>,<u'u'>,… filled in P1; placeholder for the skeleton
};

void accumulate_statistics_cpu(Stats& s, const core::CpuField& U, const core::CpuField& V,
                               const core::CpuField& W, const core::CpuField& P, const core::Subdomain& sub);
void accumulate_statistics_gpu(Stats& s, const core::GpuField& U, const core::GpuField& V,
                               const core::GpuField& W, const core::GpuField& P, const core::Subdomain& sub);

} // namespace mpmstd::post
