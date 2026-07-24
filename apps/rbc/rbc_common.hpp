#pragma once
// Shared includes, utility functions, and stats accumulator for all 4 RBC mains.

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/boundary_ops.hpp"
#include "core/fields.hpp"

#include "core/config.hpp"
#include "config/logger.hpp"
#include "boundary/problem_loader.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/pressure/solve.hpp"
#include "physics/buoyancy/buoyancy.hpp"
#include "physics/properties/properties.hpp"
#include "driver/cfl.hpp"
#include "post/statistics.hpp"
#include "post/io.hpp"
#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>


namespace rbc {
using namespace mpmstd;

inline void zero_w_wall(core::CpuField& W, const parallel::mpi::Subdomain& sub) {
  const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  real_t* w = W.data();
  if (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) w[(i * n2 + j) * n3 + h] = 0;
  if (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) w[(i * n2 + j) * n3 + (n3 - h)] = 0;
}

inline double div_max(const core::Domain& d, const core::CpuFields& f) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t *u = f[core::Var::U].data(), *v = f[core::Var::V].data(), *w = f[core::Var::W].data();
  const real_t *dx1 = d.grid.dx_ptr(Direction::X), *dx2 = d.grid.dx_ptr(Direction::Y), *dx3 = d.grid.dx_ptr(Direction::Z);
  double local = 0.0;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const double dv = (double(u[((i+1)*n2+j)*n3+k]) - double(u[(i*n2+j)*n3+k])) / dx1[i]
                        + (double(v[(i*n2+j+1)*n3+k]) - double(v[(i*n2+j)*n3+k])) / dx2[j]
                        + (double(w[(i*n2+j)*n3+k+1]) - double(w[(i*n2+j)*n3+k])) / dx3[k];
        local = std::max(local, std::fabs(dv));
      }
  double g = local;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, d.sub.topology().cart_comm());
  return g;
}

struct FlowDiag { double umax, vmax, wmax, tmax, tmin; int kwmax; };

inline FlowDiag flow_diag(const core::Domain& d, const core::CpuFields& f) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t *u = f[core::Var::U].data(), *v = f[core::Var::V].data();
  const real_t *w = f[core::Var::W].data(), *T = f[core::Var::T].data();
  double um=0, vm=0, wm=0, tx=-1e30, tn=1e30, wbest=-1; int kw=-1;
  const int kstart = d.sub.global_offset()[2];
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i*n2+j)*n3+k;
        um = std::max(um, std::fabs(double(u[c]))); vm = std::max(vm, std::fabs(double(v[c])));
        const double aw = std::fabs(double(w[c]));
        if (aw > wbest) { wbest = aw; kw = kstart + (k - h); }
        wm = std::max(wm, aw);
        tx = std::max(tx, double(T[c])); tn = std::min(tn, double(T[c]));
      }
  double in[5] = {um, vm, wm, tx, -tn}, out[5];
  MPI_Allreduce(in, out, 5, MPI_DOUBLE, MPI_MAX, d.sub.topology().cart_comm());
  return {out[0], out[1], out[2], out[3], -out[4], kw};
}

inline double center_temp(const core::Domain& d, const core::CpuFields& f) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const int nz_g = d.sub.n_global()[2], kc_g = nz_g / 2;
  const int kstart = d.sub.global_offset()[2], nz_loc = d.sub.n_interior()[2];
  const real_t* T = f[core::Var::T].data();
  double local = 0.0;
  if (kc_g >= kstart && kc_g < kstart + nz_loc) {
    const int kc = kc_g - kstart + h;
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j) local += double(T[(i*n2+j)*n3+kc]);
  }
  double g = 0.0;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_SUM, d.sub.topology().cart_comm());
  return g / (static_cast<double>(d.sub.n_global()[0]) * d.sub.n_global()[1]);
}

// ── per-run accumulators (Theta_c, Nu, z-profiles) ───────────────────────────
struct RbcStats {
  double tc_sum = 0.0; long tc_n = 0;
  std::vector<double> t_sum_z, w2_sum_z;
  double wt_sum = 0.0;
  explicit RbcStats(int nzg) : t_sum_z(nzg, 0.0), w2_sum_z(nzg, 0.0) {}
};

// ── restart helpers ───────────────────────────────────────────────────────────
struct RestartMeta { int step = 0; double time = 0.0;
                     double tc_sum = 0.0; long tc_n = 0; double wt_sum = 0.0; };

inline void write_restart(const core::CpuFields& fields, const parallel::mpi::Subdomain& sub,
                          const RbcStats& stats, int step, double time,
                          const std::string& dir, MPI_Comm comm) {
  std::error_code ec; std::filesystem::create_directories(dir, ec);
  MPI_Barrier(comm);
  // gathered (root-gather + fwrite), NOT the MPI-IO collective write_restart_cpu:
  // the matching MPI-IO read (read_restart_cpu) deadlocked on this cluster's NFS
  // at np=2/256^3 (see Report/34 addendum) — DHVC's restart write already uses
  // this same gathered path for the same reason.
  post::write_restart_gathered_cpu(fields[core::Var::U], sub, dir + "/U.bin");
  post::write_restart_gathered_cpu(fields[core::Var::V], sub, dir + "/V.bin");
  post::write_restart_gathered_cpu(fields[core::Var::W], sub, dir + "/W.bin");
  post::write_restart_gathered_cpu(fields[core::Var::T], sub, dir + "/T.bin");
  post::write_restart_gathered_cpu(fields[core::Var::P], sub, dir + "/P.bin");  // accumulated pressure (van Kan)
  int rank; MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    std::ofstream sf(dir + "/stats.txt");
    sf << std::setprecision(17)
       << "step "   << step         << "\n"
       << "time "   << time         << "\n"
       << "tc_sum " << stats.tc_sum << "\n"
       << "tc_n "   << stats.tc_n   << "\n"
       << "wt_sum " << stats.wt_sum << "\n";
    std::printf("[restart] wrote to %s (step=%d tc_n=%ld)\n", dir.c_str(), step, stats.tc_n);
  }
}

inline RestartMeta read_restart(core::CpuFields& fields, const boundary::BoundaryCondition& bc,
                                const parallel::mpi::Subdomain& sub,
                                const std::string& dir, MPI_Comm comm) {
  post::read_restart_cpu(fields[core::Var::U], sub, dir + "/U.bin");
  post::read_restart_cpu(fields[core::Var::V], sub, dir + "/V.bin");
  post::read_restart_cpu(fields[core::Var::W], sub, dir + "/W.bin");
  post::read_restart_cpu(fields[core::Var::T], sub, dir + "/T.bin");
  // P.bin (accumulated pressure, van Kan) — optional for backward compatibility
  // with pre-Report/32 restarts; absent → P stays zero (self-starting projection).
  // existence must be decided on ONE rank and broadcast: read_restart_cpu/
  // sync_field_cpu below are collective (MPI_File_read_all / halo exchange), so a
  // per-rank filesystem::exists() that disagrees across ranks (e.g. NFS attribute-
  // cache skew) makes some ranks enter the collective while others skip it —
  // deadlock. See Report/34 sec. "restart read hang".
  int has_p_bin = 0;
  {
    int rank; MPI_Comm_rank(comm, &rank);
    if (rank == 0) has_p_bin = std::filesystem::exists(dir + "/P.bin") ? 1 : 0;
    MPI_Bcast(&has_p_bin, 1, MPI_INT, 0, comm);
  }
  if (has_p_bin) {
    post::read_restart_cpu(fields[core::Var::P], sub, dir + "/P.bin");
    core::sync_field_cpu(fields, core::Var::P, bc, sub);
  }
  core::sync_field_cpu(fields, core::Var::U, bc, sub);
  core::sync_field_cpu(fields, core::Var::V, bc, sub);
  core::sync_field_cpu(fields, core::Var::W, bc, sub);
  core::sync_field_cpu(fields, core::Var::T, bc, sub);
  RestartMeta m;
  int rank; MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    std::ifstream sf(dir + "/stats.txt");
    std::string key; double val;
    while (sf >> key >> val) {
      if      (key == "step")    m.step    = (int)val;
      else if (key == "time")    m.time    = val;
      else if (key == "tc_sum")  m.tc_sum  = val;
      else if (key == "tc_n")    m.tc_n    = (long)val;
      else if (key == "wt_sum")  m.wt_sum  = val;
    }
    std::printf("[restart] loaded step=%d t=%.3g tc_n=%ld\n", m.step, m.time, m.tc_n);
  }
  MPI_Bcast(&m.step,   1, MPI_INT,    0, comm);
  MPI_Bcast(&m.time,   1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&m.tc_sum, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&m.tc_n,   1, MPI_LONG,   0, comm);
  MPI_Bcast(&m.wt_sum, 1, MPI_DOUBLE, 0, comm);
  return m;
}

inline void apply_stats_from_restart(RbcStats& stats, const RestartMeta& m) {
  stats.tc_sum = m.tc_sum; stats.tc_n = m.tc_n; stats.wt_sum = m.wt_sum;
}

inline void accumulate(RbcStats& s, const core::Domain& d, const parallel::mpi::Subdomain& sub,
                       const core::CpuFields& f, const real_t* dx3_z, int nstat_start, int step) {
  if (step < nstat_start) return;
  s.tc_sum += center_temp(d, f);
  ++s.tc_n;
  const auto nt = sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2], h=kHaloWidth;
  const int koff = sub.global_offset()[2], nzl = sub.n_interior()[2];
  const real_t *Tp = f[core::Var::T].data(), *Wp = f[core::Var::W].data();
  double wt_loc = 0.0;
  for (int i = h; i < n1-h; ++i)
    for (int j = h; j < n2-h; ++j)
      for (int kk = 0; kk < nzl; ++kk) {
        const int k = kk + h;
        const double tv = double(Tp[(i*n2+j)*n3+k]), wv = double(Wp[(i*n2+j)*n3+k]);
        s.t_sum_z[koff+kk] += tv; s.w2_sum_z[koff+kk] += wv*wv;
        wt_loc += wv * tv * double(dx3_z[k]);
      }
  double wt_g = 0.0;
  MPI_Allreduce(&wt_loc, &wt_g, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  s.wt_sum += wt_g;
}

inline void print_step(const RbcStats& s, const core::Domain& d, const core::CpuFields& f,
                       int step, double time, real_t dt, double alpha_T, bool is_root) {
  const double dv = div_max(d, f);
  const double tc = (s.tc_n > 0) ? s.tc_sum / s.tc_n : center_temp(d, f);
  const FlowDiag fd = flow_diag(d, f);
  if (!is_root) return;
  std::printf("step=%d t=%.3g dt=%.3g div=%.2e Theta_c=%.4f | umax=%.3g vmax=%.3g wmax=%.3g(k=%d) T=[%.3f,%.3f]\n",
              step, time, double(dt), dv, tc, fd.umax, fd.vmax, fd.wmax, fd.kwmax, fd.tmin, fd.tmax);
  if (s.tc_n > 0) {
    const int nxg = d.sub.n_global()[0], nyg = d.sub.n_global()[1];
    const double wt_now = s.wt_sum / (double(nxg)*nyg) / s.tc_n;
    std::printf("  [Nu] running: Nu=%.4f (tc_n=%ld)\n", 1.0 + wt_now/alpha_T, s.tc_n);
  }
}

inline void print_final(RbcStats& s, const parallel::mpi::Subdomain& sub,
                        double Ra, double Pr, double alpha_T, bool is_root, const char* tag) {
  const double theta_c_out = (s.tc_n > 0) ? s.tc_sum / s.tc_n : 0.0;
  if (is_root) std::printf("[%s] done: Ra=%.2e Pr=%.4g  Theta_c=%.5f\n", tag, Ra, Pr, theta_c_out);
  if (s.tc_n <= 0) return;
  const int nzg = static_cast<int>(s.t_sum_z.size());
  const int nxg = sub.n_global()[0], nyg = sub.n_global()[1];
  std::vector<double> t_z_g(nzg, 0.0), w2_z_g(nzg, 0.0);
  MPI_Allreduce(s.t_sum_z.data(),  t_z_g.data(),  nzg, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  MPI_Allreduce(s.w2_sum_z.data(), w2_z_g.data(), nzg, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  const double inv_nxy = 1.0 / (double(nxg)*nyg), inv_n = 1.0 / s.tc_n;
  const double Nu = 1.0 + s.wt_sum * inv_nxy * inv_n / alpha_T;
  if (!is_root) return;
  std::printf("[%s] Nu=%.4f  (wt_mean=%.6f  alpha_T=%.4e  tc_n=%ld)\n",
              tag, Nu, s.wt_sum * inv_nxy * inv_n, alpha_T, s.tc_n);
  std::printf("[%s] k  T_mean  W_rms\n", tag);
  for (int k = 0; k < nzg; ++k)
    std::printf("  %3d  %+.5f  %.5f\n", k, t_z_g[k]*inv_nxy*inv_n,
                std::sqrt(std::max(0.0, w2_z_g[k]*inv_nxy*inv_n)));
}

} // namespace rbc
