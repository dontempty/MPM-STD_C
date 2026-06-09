// apps/channel — isothermal turbulent channel (rev.2 §7 case A), free-function
// recipe on CpuField. Faithful port of the validated apps/channel time loop:
//   cfl -> assemble_momentum_const_visc -> solve_momentum (ADI+block coupling) ->
//   update_velocity -> channel forcing (dPdx + mass-flow) -> halo/ghost ->
//   solve_pressure (div+Poisson+project) -> statistics.
// Isothermal: NO T / mu declared (zero compute, zero memory).

#include "core/field_cpu.hpp"
#include "core/grid.hpp"
#include "core/boundary.hpp"
#include "core/mpi_topology.hpp"
#include "core/halo.hpp"
#include "core/boundary_ops.hpp"
#include "core/system.hpp"
#include "core/config.hpp"

#include "config/logger.hpp"
#include "boundary/problem_loader.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"

#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/solve.hpp"
#include "physics/forcing/forcing.hpp"
#include "driver/cfl.hpp"
#include "post/statistics.hpp"
#include "post/io.hpp"

#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

using namespace mpmstd;

namespace {

// Enforce W=0 on global z-wall faces (MAC stagger; matches the validated loop).
void zero_w_wall(core::CpuField& W, const core::Subdomain& sub) {
  const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  real_t* w = W.data();
  if (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) w[(i * n2 + j) * n3 + h] = 0;
  if (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) w[(i * n2 + j) * n3 + (n3 - h)] = 0;
}

// Global max |div(U*)| (solver-health monitor).
double div_max(const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
               const core::Grid& g, const core::Subdomain& sub) {
  const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = U.data(); const real_t* v = V.data(); const real_t* w = W.data();
  const real_t* dx1 = g.dx_ptr(Direction::X), *dx2 = g.dx_ptr(Direction::Y), *dx3 = g.dx_ptr(Direction::Z);
  double local = 0.0;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const double d = (double(u[((i + 1) * n2 + j) * n3 + k]) - double(u[(i * n2 + j) * n3 + k])) / dx1[i]
                       + (double(v[(i * n2 + j + 1) * n3 + k]) - double(v[(i * n2 + j) * n3 + k])) / dx2[j]
                       + (double(w[(i * n2 + j) * n3 + k + 1]) - double(w[(i * n2 + j) * n3 + k])) / dx3[k];
        local = std::max(local, std::fabs(d));
      }
  double global = local;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, sub.topology().cart_comm());
  return global;
}

} // anonymous namespace

int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/channel/input.toml";
  auto cfg = config::Config::load(input_path);

  // ── topology / subdomain / grid / BC / TDMA ──────────────────────────────
  auto is_per = [](const std::string& s) { return s == "periodic"; };
  std::array<int, 3>  dims     = {cfg.get<int>("mpi", "np1"), cfg.get<int>("mpi", "np2"), cfg.get<int>("mpi", "np3")};
  std::array<bool, 3> periodic = {is_per(cfg.get_or<std::string>("topology", "x", "periodic")),
                                  is_per(cfg.get_or<std::string>("topology", "y", "periodic")),
                                  is_per(cfg.get_or<std::string>("topology", "z", "wall"))};
  core::MpiTopology topo(mpi, dims, periodic);
  std::array<int, 3> n_global = {cfg.get<int>("mesh", "n1m"), cfg.get<int>("mesh", "n2m"), cfg.get<int>("mesh", "n3m")};
  core::Subdomain sub(topo, n_global);

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a) {
    const int    is_uni = cfg.get_or<int>("uniform_mesh", "uniform" + std::to_string(a + 1), 1);
    const double gamma  = cfg.get_or<double>("mesh_stretch", "gamma" + std::to_string(a + 1), 0.0);
    const double L      = cfg.get<double>("domain", std::string("L") + char('x' + a));
    axes[a] = {n_global[a], L, is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
  }
  core::Grid g(sub, axes);

  core::Boundary problem = boundary::load_problem_from_config(cfg);
  problem.validate();
  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);

  // ── fields (isothermal: no T/mu) + systems ───────────────────────────────
  core::CpuField U(sub, "U"), V(sub, "V"), W(sub, "W"), P(sub, "P");
  core::CpuField dU(sub, "dU"), dV(sub, "dV"), dW(sub, "dW");
  core::MomentumSystem mom;
  core::PressureSystem poi;
  post::Stats stats;
  post::init_statistics_cpu(stats, g, sub);

  // ── physics params + forcing ──────────────────────────────────────────────
  const double nu = cfg.has("physics", "nu") ? cfg.get<double>("physics", "nu")
                    : std::sqrt(cfg.get<double>("physics", "Pr") / cfg.get<double>("physics", "Ra"));
  const std::string fmode = cfg.get_or<std::string>("channel_forcing", "mode", "pressure_gradient");
  const bool   mass_flow  = (fmode == "mass_flow");
  const double target_Ub  = cfg.get_or<double>("channel_forcing", "target_bulk_velocity", 1.0);
  const double target_dpdx= cfg.get_or<double>("channel_forcing", "target_dPdx",
                              cfg.get_or<double>("source", "U_force", 3.0 * nu));
  const double total_vol  = physics::channel_total_volume_cpu(g, sub);
  double dpdx = mass_flow ? -(3.0 * nu) : -target_dpdx;   // -ve drives +x; F_lam=3nu seed

  // ── time / IO config ──────────────────────────────────────────────────────
  const double t_end    = cfg.get_or<double>("time", "t_end", 1000.0);
  const real_t dt_cap   = static_cast<real_t>(cfg.get_or<double>("time", "dt_start", 0.05));
  const real_t max_cfl  = static_cast<real_t>(cfg.get_or<double>("time", "MaxCFL", 1.0));
  const int    n_steps  = cfg.get_or<int>("time", "n_steps", 1000);
  const std::string dir_restart = cfg.get_or<std::string>("output", "dir_restart", "apps/channel/restart_in/");
  const std::string dir_stats   = cfg.get_or<std::string>("output", "dir_stats", "apps/channel/statistics/");
  const int  nmonitor    = cfg.get_or<int>("output", "nmonitor", 100);
  const int  nstat_start = cfg.get_or<int>("output", "nstat_start", 0);
  const int  nstat_int   = std::max(1, cfg.get_or<int>("output", "nstat", 1));
  const int  nout_stats  = cfg.get_or<int>("output", "nout_stats", 10000);
  const bool out_stats   = cfg.get_or<int>("output", "out_stats", 1) != 0;
  const bool continue_in = cfg.get_or<int>("output", "continue_in", 1) != 0;

  if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
  MPI_Barrier(topo.cart_comm());

  // ── initial condition: load frozen turbulent restart ──────────────────────
  U.fill(0); V.fill(0); W.fill(0); P.fill(0);
  double time = cfg.get_or<double>("time", "t_start", 0.0);
  int    step = 0;
  if (continue_in) {
    if (mpi.is_root()) std::printf("[channel] restart from '%s'\n", dir_restart.c_str());
    post::read_restart_cpu(U, sub, dir_restart + "U.bin");
    post::read_restart_cpu(V, sub, dir_restart + "V.bin");
    post::read_restart_cpu(W, sub, dir_restart + "W.bin");
    post::read_restart_cpu(P, sub, dir_restart + "P.bin");
    if (mpi.is_root()) {
      FILE* mf = std::fopen((dir_restart + "meta.txt").c_str(), "r");
      if (mf) { double dtd, dpd; int st;
        if (std::fscanf(mf, "%d %lf %lf %lf", &st, &time, &dtd, &dpd) == 4) { step = st; dpdx = dpd; }
        std::fclose(mf); }
    }
    MPI_Bcast(&time, 1, MPI_DOUBLE, 0, topo.cart_comm());
    MPI_Bcast(&dpdx, 1, MPI_DOUBLE, 0, topo.cart_comm());
    MPI_Bcast(&step, 1, MPI_INT, 0, topo.cart_comm());
  }
  zero_w_wall(W, sub);
  core::sync_field_cpu(U, problem.U, sub); core::sync_field_cpu(V, problem.V, sub);
  core::sync_field_cpu(W, problem.W, sub); core::sync_field_cpu(P, problem.P, sub);

  if (mpi.is_root())
    std::printf("[channel] start step=%d time=%.4g  nu=%.4e  mode=%s  np=%dx%dx%d  n=%dx%dx%d\n",
                step, time, nu, fmode.c_str(), dims[0], dims[1], dims[2], n_global[0], n_global[1], n_global[2]);

  // ── time loop (rev.2 §7 recipe) ────────────────────────────────────────────
  const int step0 = step;
  while (time < t_end && (step - step0) < n_steps) {
    // (1) CFL-limited time step
    real_t dt = driver::compute_cfl_dt_cpu(U, V, W, g, sub, max_cfl, dt_cap);

    // (2) momentum: explicit RHS → 3-sweep ADI + block coupling → U += dU
    equation::assemble_momentum_const_visc_cpu(mom, U, V, W, g, static_cast<real_t>(nu), dt);
    equation::solve_momentum_cpu(mom, U, V, W, dU, dV, dW, g, problem, *tdma, sub, static_cast<real_t>(nu), dt);
    equation::update_velocity_cpu(U, V, W, dU, dV, dW);
    core::sync_field_cpu(V, problem.V, sub);    // update_velocity changed interiors → refresh V,W
    core::sync_field_cpu(W, problem.W, sub);

    // (3) channel forcing: mean pressure gradient + (mass-flow) bulk correction
    physics::apply_body_force_cpu(U, static_cast<real_t>(dpdx), dt);
    if (mass_flow) physics::apply_mass_flow_correction_cpu(U, target_Ub, g, sub, dt, total_vol, dpdx);
    core::sync_field_cpu(U, problem.U, sub);     // U changed by forcing → refresh
    zero_w_wall(W, sub);                          // MAC: enforce W=0 on z-wall faces

    // (4) pressure: div(U*) → Poisson → project U,V,W divergence-free (P updated)
    equation::solve_pressure_cpu(poi, dt, U, V, W, P, g, problem, *tdma, sub);

    time += static_cast<double>(dt);
    ++step;

    // (5) post: statistics + monitor

    if (out_stats && (step - step0) >= nstat_start && ((step - step0) % nstat_int == 0))
      post::accumulate_statistics_cpu(stats, U, V, W, P, sub);

    if (step % nmonitor == 0) {
      const double dv = div_max(U, V, W, g, sub);
      const double ub = physics::channel_bulk_velocity_cpu(U, g, sub, total_vol);
      if (mpi.is_root())
        std::printf("step=%d time=%.4g dt=%.3g div=%.3e Ub=%.4f dPdx=%.4e\n",
                    step, time, double(dt), dv, ub, dpdx);
    }
    if (out_stats && step % nout_stats == 0)
      post::write_statistics_cpu(stats, dir_stats + "stats_" + std::to_string(step) + ".dat",
                                 step, nu, sub, g);
  }

  if (out_stats)
    post::write_statistics_cpu(stats, dir_stats + "stats_final.dat", step, nu, sub, g);
  if (mpi.is_root()) std::printf("[channel] done step=%d time=%.4g\n", step, time);
  return 0;
}
