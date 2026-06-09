#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/mpi_topology.hpp"

#include <string>
#include <vector>

namespace mpmstd::post {

// Time-averaged z-profile channel statistics (Welford). ⚠ The xy-plane average
// uses the GLOBAL nx*ny normalization — dividing by per-rank n_interior is the
// 16x bug (see memory channel-wss-diagnostic). Each rank owns distinct z-slabs,
// so write() MPI_SUM-reduces the per-slab locals with no extra division.
struct Stats {
  int  nz_global = 0, nz_local = 0, kstart = 0;
  long n = 0;
  std::vector<double> U_m, U2_m, V_m, V2_m, Wc_m, Wc2_m, UWc_m, P_m;
  std::vector<double> zc_global;
};

void init_statistics_cpu(Stats& s, const core::Grid& grid, const core::Subdomain& sub);

void accumulate_statistics_cpu(Stats& s, const core::CpuField& U, const core::CpuField& V,
                               const core::CpuField& W, const core::CpuField& P, const core::Subdomain& sub);

void write_statistics_cpu(const Stats& s, const std::string& path, int step, double nu,
                          const core::Subdomain& sub, const core::Grid& grid);

void accumulate_statistics_gpu(Stats& s, const core::GpuField& U, const core::GpuField& V,
                               const core::GpuField& W, const core::GpuField& P, const core::Subdomain& sub);

} // namespace mpmstd::post
