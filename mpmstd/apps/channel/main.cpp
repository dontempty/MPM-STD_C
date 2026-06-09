// apps/channel — isothermal turbulent channel (rev.2 §7 recipe), structural
// redesign: the loop drives a Domain (geometry/parallel/tdma) + a BoundaryCondition
// + a Fields container (the in-use physical variables + constants, registered
// before the loop). Momentum owns its own increments. No per-field arg clutter,
// no pointer derefs in the loop.

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/boundary_ops.hpp"    // sync_field_cpu
#include "core/variables.hpp"
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

// Enforce W=0 on global z-wall faces (MAC stagger).
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
double div_max(const core::Domain& d, const core::CpuFields& fields) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = fields[core::Var::U].data(); const real_t* v = fields[core::Var::V].data(); const real_t* w = fields[core::Var::W].data();
  const real_t* dx1 = d.grid.dx_ptr(Direction::X), *dx2 = d.grid.dx_ptr(Direction::Y), *dx3 = d.grid.dx_ptr(Direction::Z);
  double local = 0.0;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const double dv = (double(u[((i + 1) * n2 + j) * n3 + k]) - double(u[(i * n2 + j) * n3 + k])) / dx1[i]
                        + (double(v[(i * n2 + j + 1) * n3 + k]) - double(v[(i * n2 + j) * n3 + k])) / dx2[j]
                        + (double(w[(i * n2 + j) * n3 + k + 1]) - double(w[(i * n2 + j) * n3 + k])) / dx3[k];
        local = std::max(local, std::fabs(dv));
      }
  double grid = local;
  MPI_Allreduce(&local, &grid, 1, MPI_DOUBLE, MPI_MAX, d.sub.topology().cart_comm());
  return grid;
}

} // anonymous namespace

int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/channel/input.toml";
  auto cfg = config::Config::load(input_path);

  // ── preprocessing: topology / subdomain / grid / BC / tdma ────────────────
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
  core::Grid grid(sub, axes);
  core::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
  bc.validate();
  auto  tdma_owner = linear_solver::tdma::TdmaRegistry::make_default(topo);
  auto& tdma       = *tdma_owner;                       // unwrap once — no `*` in the loop

  core::Domain domain{grid, topo, sub, tdma};                 // the "where/how" context

  // ── variables: register what this (isothermal) case uses, before the loop ─
  core::CpuFields fields;
  fields.add(core::Var::U, sub); fields.add(core::Var::V, sub);
  fields.add(core::Var::W, sub); fields.add(core::Var::P, sub);

  // ── physics params + forcing ──────────────────────────────────────────────
  const double nu = cfg.has("physics", "nu") ? cfg.get<double>("physics", "nu")
                    : std::sqrt(cfg.get<double>("physics", "Pr") / cfg.get<double>("physics", "Ra"));
  fields.add_constant(core::Const::nu, static_cast<real_t>(nu));
  const std::string fmode = cfg.get_or<std::string>("channel_forcing", "mode", "pressure_gradient");
  const bool   mass_flow  = (fmode == "mass_flow");
  const double target_Ub  = cfg.get_or<double>("channel_forcing", "target_bulk_velocity", 1.0);
  const double target_dpdx= cfg.get_or<double>("channel_forcing", "target_dPdx", cfg.get_or<double>("source", "U_force", 3.0 * nu));
  const double total_vol  = physics::channel_total_volume_cpu(domain);
  double dpdx = mass_flow ? -(3.0 * nu) : -target_dpdx;

  // ── solver state (momentum owns its increments) ───────────────────────────
  core::CpuMomentumSystem momentum(sub);
  core::PressureSystem    pressure;
  post::Stats             stats;
  post::init_statistics_cpu(stats, domain);

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

  // ── initial condition: load frozen turbulent restart into Fields ──────────
  double time = cfg.get_or<double>("time", "t_start", 0.0);
  int    step = 0;
  if (continue_in) {
    if (mpi.is_root()) std::printf("[channel] restart from '%s'\n", dir_restart.c_str());
    post::read_restart_cpu(fields[core::Var::U], sub, dir_restart + "U.bin");
    post::read_restart_cpu(fields[core::Var::V], sub, dir_restart + "V.bin");
    post::read_restart_cpu(fields[core::Var::W], sub, dir_restart + "W.bin");
    post::read_restart_cpu(fields[core::Var::P], sub, dir_restart + "P.bin");
    if (mpi.is_root()) {
      FILE* mf = std::fopen((dir_restart + "meta.txt").c_str(), "r");
      if (mf) { double dtd, dpd; int st;
        if (std::fscanf(mf, "%d %lf %lf %lf", &st, &time, &dtd, &dpd) == 4) { step = st; dpdx = dpd; }
        std::fclose(mf); }
    }
    MPI_Bcast(&time, 1, MPI_DOUBLE, 0, topo.cart_comm());
    MPI_Bcast(&dpdx, 1, MPI_DOUBLE, 0, topo.cart_comm());
    MPI_Bcast(&step, 1, MPI_INT,    0, topo.cart_comm());
  }
  zero_w_wall(fields[core::Var::W], sub);
  core::sync_field_cpu(fields, core::Var::U, bc, sub); core::sync_field_cpu(fields, core::Var::V, bc, sub);
  core::sync_field_cpu(fields, core::Var::W, bc, sub); core::sync_field_cpu(fields, core::Var::P, bc, sub);

  if (mpi.is_root())
    std::printf("[channel] start step=%d time=%.4g  nu=%.4e  mode=%s  np=%dx%dx%d  n=%dx%dx%d\n",
                step, time, nu, fmode.c_str(), dims[0], dims[1], dims[2], n_global[0], n_global[1], n_global[2]);

  // ── time loop (rev.2 §7 recipe) ────────────────────────────────────────────
  const int step0 = step;
  while (time < t_end && (step - step0) < n_steps) {
    const real_t dt = driver::compute_cfl_dt_cpu(domain, fields, max_cfl, dt_cap);

    // (2) momentum: RHS → 3-sweep ADI + block coupling → U += dU
    equation::assemble_momentum_const_visc_cpu(domain, fields, momentum, dt);
    equation::solve_momentum_cpu(domain, bc, fields, momentum, dt);
    equation::update_velocity_cpu(fields, momentum);
    core::sync_field_cpu(fields, core::Var::V, bc, sub);
    core::sync_field_cpu(fields, core::Var::W, bc, sub);

    // (3) channel forcing: mean dP/dx + (mass-flow) bulk correction
    physics::apply_body_force_cpu(fields, static_cast<real_t>(dpdx), dt);
    if (mass_flow) physics::apply_mass_flow_correction_cpu(domain, fields, target_Ub, dt, total_vol, dpdx);
    core::sync_field_cpu(fields, core::Var::U, bc, sub);
    zero_w_wall(fields[core::Var::W], sub);

    // (4) pressure: div → Poisson → project U,V,W (P updated)
    equation::solve_pressure_cpu(domain, bc, fields, pressure, dt);

    time += static_cast<double>(dt);
    ++step;

    // (5) post: statistics + monitor
    if (out_stats && (step - step0) >= nstat_start && ((step - step0) % nstat_int == 0))
      post::accumulate_statistics_cpu(stats, domain, fields);
    if (step % nmonitor == 0) {
      const double dv = div_max(domain, fields);
      const double ub = physics::channel_bulk_velocity_cpu(domain, fields, total_vol);
      if (mpi.is_root())
        std::printf("step=%d time=%.4g dt=%.3g div=%.3e Ub=%.4f dPdx=%.4e\n", step, time, double(dt), dv, ub, dpdx);
    }
    if (out_stats && step % nout_stats == 0)
      post::write_statistics_cpu(stats, dir_stats + "stats_" + std::to_string(step) + ".dat", step, nu, domain);
  }

  if (out_stats) post::write_statistics_cpu(stats, dir_stats + "stats_final.dat", step, nu, domain);
  if (mpi.is_root()) std::printf("[channel] done step=%d time=%.4g\n", step, time);
  return 0;
}
