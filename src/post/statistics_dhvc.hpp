#pragma once

#include "core/fields.hpp"        // core::GpuFields, real_t
#include "core/gpu/gpu_grid.hpp"  // core::GpuGridArrays
#include "core/domain.hpp"        // core::Domain, parallel::mpi::Subdomain, grid::Grid

#include <mpi.h>
#include <memory>
#include <string>

namespace mpmstd::post {

// Config for the DHVC GPU statistics object (values from [output]/[physics]).
// Time-based windows: accumulate a dt-weighted (true time-average) sample every
// step once physical time reaches t_stat_start; write a cumulative-average
// profile every t_stat_out.
struct DhvcStatisticsConfig {
  int         nmonitor     = 500;
  double      t_stat_start = 0.0, t_stat_out = 0.0;
  std::string dir_stats;
  double      Ra = 0.0, nu = 0.0, alpha_T = 0.0;
};

// DHVC statistics infrastructure lifted out of main_gpu.cu: device reductions
// (z-profiles, flow-diag, KE/Nu/theta_rms, Re_delta*), monitor.csv, and the
// Tecplot stats_t*.dat writer (via post::write_statistics_cpu). Numerics and
// file formats are unchanged. GPU-only (included solely from .cu). The wall-normal
// W-zeroing kernel stays in main (a physics guard, not a statistic).
class DhvcStatistics {
public:
  DhvcStatistics(const parallel::mpi::Subdomain& sub, const grid::Grid& grid,
                 const core::GpuGridArrays& gg, const core::Domain& domain,
                 const DhvcStatisticsConfig& cfg, MPI_Comm comm, bool is_root);
  ~DhvcStatistics();
  DhvcStatistics(const DhvcStatistics&)            = delete;
  DhvcStatistics& operator=(const DhvcStatistics&) = delete;

  void accumulate_if_due(double time, real_t dt, const core::GpuFields& gf);        // dt-weighted sample
  void monitor_if_due(int step, double time, real_t dt, const core::GpuFields& gf,
                      double cum_wall_s, double recent_ms_step);                    // printf + monitor.csv
  void write_profile_if_due(double time, int step);                                // stats_t<t>.dat
  double final_re_delta_star() const;   // post::compute_re_delta_star_cpu on the accumulated profile

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mpmstd::post
