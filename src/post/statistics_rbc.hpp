#pragma once

#include "core/fields.hpp"        // core::GpuFields, real_t
#include "core/gpu/gpu_grid.hpp"  // core::GpuGridArrays
#include "core/domain.hpp"        // parallel::mpi::Subdomain, grid::Grid
#include "post/restart.hpp"       // post::RestartState

#include <mpi.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mpmstd::post {

// Configuration for the RBC GPU statistics object (values read from [output]/
// [physics] in main). Time-based windows: accumulate once physical time reaches
// t_stat_start, write a cumulative-average profile every t_stat_out.
struct RbcStatisticsConfig {
  double      t_stat_start = 0.0;
  double      t_stat_out   = 0.0;
  int         nmonitor     = 200;
  std::string dir_stats;                 // monitor.csv + stats_t*.dat location
  double      Ra = 0.0, Pr = 0.0, DeltaT = 0.0, alpha_T = 0.0;
};

// RBC statistics infrastructure lifted out of main_gpu_nob.cu: device-side
// reductions (Theta_c, Nu, z-profiles, flow diagnostics), monitor.csv, and the
// stats_t*.dat / stats_final.dat writers. Numerics and file formats are
// unchanged — this only relocates the ~344 lines and hides the CUDA buffers
// behind a PImpl. GPU-only (included solely from .cu). The T-clip kernel stays
// in main (it is a physics guard, not a statistic); only its activation counter
// is owned here.
class RbcStatistics {
public:
  RbcStatistics(const parallel::mpi::Subdomain& sub, const core::GpuGridArrays& gg,
                const grid::Grid& grid, const RbcStatisticsConfig& cfg,
                MPI_Comm comm, bool is_root);
  ~RbcStatistics();
  RbcStatistics(const RbcStatistics&)            = delete;
  RbcStatistics& operator=(const RbcStatistics&) = delete;

  void accumulate_if_due(double time, const core::GpuFields& gf);                   // [9a]
  void monitor_if_due(int step, double time, real_t dt, const core::GpuFields& gf); // [9b]
  void write_profile_if_due(double time, int step);                                 // [9c]
  void finalize(int step);                                                          // stats_final + print

  // Restart bridging: {tc_sum, tc_n, wt_sum}.
  std::vector<std::pair<std::string, double>> export_extras() const;
  void import_extras(const RestartState& state, bool stats_reset);

  // Counter that k_clip_T_gpu (in main) atomically bumps; monitor_if_due reports it.
  unsigned long long* clip_counter();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mpmstd::post
