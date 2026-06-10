// apps/dhvc — OB differentially-heated vertical convection (PaScaL_TCS Fig 7).
//
// Axis map (matches the validated src/ DHVC convention; reuses the x,y-periodic
// FFT Poisson unchanged):
//   x = streamwise / vertical / gravity  (periodic, Lx = 8H)   ← buoyancy on U
//   y = spanwise                         (periodic, Ly = 4H)
//   z = wall-normal (the differentially heated walls)          (wall, Lz = H)
//       hot wall θ=+0.5 at z=0, cold wall θ=−0.5 at z=Lz; no-slip U,V,W.
// OB free-fall nondim: nu = √(Pr/Ra), alpha = 1/√(Ra·Pr), buoyancy F = θ on U.
// Recipe each step: energy → momentum(+buoyancy) → pressure → project.
// Diagnostic: Re_δ* = U_max·δ*/ν vs Ng et al. (0.23·Ra^0.28).

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/boundary_ops.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"
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
#include "physics/forcing/forcing.hpp"   // channel_total_volume_cpu, apply_mass_flow_correction_cpu (zero-net-flux)
#include "driver/cfl.hpp"
#include "post/statistics.hpp"

#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

using namespace mpmstd;

namespace {

void zero_w_wall(core::CpuField& W, const core::Subdomain& sub) {
  const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  real_t* w = W.data();
  if (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) w[(i * n2 + j) * n3 + h] = 0;
  if (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL)
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) w[(i * n2 + j) * n3 + (n3 - h)] = 0;
}

double div_max(const core::Domain& d, const core::CpuFields& f) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = f[core::Var::U].data(); const real_t* v = f[core::Var::V].data(); const real_t* w = f[core::Var::W].data();
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
  double g = local;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, d.sub.topology().cart_comm());
  return g;
}

} // anonymous namespace

int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/dhvc/input.toml";
  auto cfg = config::Config::load(input_path);

  // ── topology / subdomain / grid / BC / tdma ───────────────────────────────
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
  auto& tdma       = *tdma_owner;
  core::Domain domain{grid, topo, sub, tdma};

  // ── variables: U,V,W,P + T (OB scalar) + constants nu, alpha ──────────────
  core::CpuFields fields;
  fields.add(core::Var::U, sub); fields.add(core::Var::V, sub);
  fields.add(core::Var::W, sub); fields.add(core::Var::P, sub);
  fields.add(core::Var::T, sub);

  const double Ra      = cfg.get<double>("physics", "Ra");
  const double Pr      = cfg.get<double>("physics", "Pr");
  const double nu      = std::sqrt(Pr / Ra);
  const double alpha_T = 1.0 / std::sqrt(Ra * Pr);
  fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  // ── solver state ──────────────────────────────────────────────────────────
  core::CpuMomentumSystem momentum(sub);
  core::CpuScalarSystem   scalar_system(sub);
  core::PressureSystem    pressure;
  post::Stats             stats;
  post::init_statistics_cpu(stats, domain);

  // OB Boussinesq: gravity along x (streamwise), F = θ, coeff = 1.
  physics::BuoyancyParams buoy; buoy.gravity_axis = 0; buoy.coeff = 1; buoy.T_ref = 0;

  // ── time / IO ──────────────────────────────────────────────────────────────
  const double t_end    = cfg.get_or<double>("time", "t_end", 400.0);
  const real_t dt_cap   = static_cast<real_t>(cfg.get_or<double>("time", "dt_start", 0.02));
  const real_t max_cfl  = static_cast<real_t>(cfg.get_or<double>("time", "MaxCFL", 0.5));
  const int    n_steps  = cfg.get_or<int>("time", "n_steps", 200000);
  const int    nmonitor    = cfg.get_or<int>("output", "nmonitor", 200);
  const int    nstat_start = cfg.get_or<int>("output", "nstat_start", 20000);
  const int    nstat_int   = std::max(1, cfg.get_or<int>("output", "nstat", 10));
  const std::string dir_stats = cfg.get_or<std::string>("output", "dir_stats", "apps/dhvc/statistics/");
  if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
  MPI_Barrier(topo.cart_comm());

  // ── initial condition: analytical DHVC base U(z) + linear conductive T(z)
  //    + small multi-mode perturbation to seed 3D convective instability. ────
  {
    const double Lx = cfg.get<double>("domain", "Lx");
    const double Ly = cfg.get<double>("domain", "Ly");
    const double Lz = cfg.get<double>("domain", "Lz");
    const double T_lo = bc.T.face(Direction::Z, Side::Minus).value;   // hot (+0.5) at z=0
    const double T_hi = bc.T.face(Direction::Z, Side::Plus ).value;   // cold (−0.5) at z=Lz
    const double amp  = cfg.get_or<double>("init", "perturb", 0.05);
    const auto& xc = grid.xc(Direction::X); const auto& yc = grid.xc(Direction::Y); const auto& zc = grid.xc(Direction::Z);
    const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
    const auto off = sub.global_offset();
    real_t* U = fields[core::Var::U].data(); real_t* V = fields[core::Var::V].data();
    real_t* W = fields[core::Var::W].data(); real_t* T = fields[core::Var::T].data();
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        for (int k = 0; k < n3; ++k) {
          const int c = (i * n2 + j) * n3 + k;
          const double z = zc[k], x = xc[i], y = yc[j];
          // analytical closed-channel base shear flow (zero net flux):
          U[c] = static_cast<real_t>(z * (2.0 * z / Lz - 1.0) * (z / Lz - 1.0) / (12.0 * nu));
          T[c] = static_cast<real_t>(T_lo + (T_hi - T_lo) * (z / Lz));
          // perturbation: streamwise/spanwise modes × wall-normal envelope
          const double env = std::sin(M_PI * z / Lz);
          const double pert = amp * env * (std::sin(2.0 * M_PI * x / Lx) * std::cos(2.0 * M_PI * y / Ly)
                                         + std::sin(4.0 * M_PI * x / Lx) * std::sin(2.0 * M_PI * y / Ly));
          V[c] += static_cast<real_t>(pert);
          W[c] += static_cast<real_t>(0.5 * amp * env * std::sin(2.0 * M_PI * x / Lx));
          (void)off;
        }
  }
  zero_w_wall(fields[core::Var::W], sub);
  core::sync_field_cpu(fields, core::Var::U, bc, sub); core::sync_field_cpu(fields, core::Var::V, bc, sub);
  core::sync_field_cpu(fields, core::Var::W, bc, sub); core::sync_field_cpu(fields, core::Var::P, bc, sub);
  core::sync_field_cpu(fields, core::Var::T, bc, sub);

  if (mpi.is_root())
    std::printf("[dhvc] Ra=%.2e Pr=%.3g  nu=%.4e alpha=%.4e  n=%dx%dx%d (x stream, y span, z wall)\n",
                Ra, Pr, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

  // Closed DHVC has ZERO net streamwise flux: the periodic streamwise direction
  // has no pressure gradient to balance the net buoyancy force (∫θ dz drifts off
  // zero once symmetry breaks), so a spurious net flow would otherwise accumulate
  // and swamp the antisymmetric shear jet. Enforce <U>=0 each step.
  const double total_vol = physics::channel_total_volume_cpu(domain);

  // ── time loop ──────────────────────────────────────────────────────────────
  double time = 0.0; int step = 0;
  while (time < t_end && step < n_steps) {
    const real_t dt = driver::compute_cfl_dt_cpu(domain, fields, max_cfl, dt_cap);

    // (1) energy: T^n → T^{n+1}
    equation::assemble_scalar_const_diff_cpu(domain, fields, scalar_system, dt);
    equation::solve_scalar_cpu(domain, bc, fields, scalar_system, dt);
    core::sync_field_cpu(fields, core::Var::T, bc, sub);

    // (2) momentum predictor + OB buoyancy (F=θ on U) composed in main
    equation::assemble_momentum_const_visc_cpu(domain, fields, momentum, dt);
    physics::add_buoyancy_force_cpu(momentum, fields[core::Var::T], buoy, dt);
    equation::solve_momentum_cpu(domain, bc, fields, momentum, dt);
    equation::update_velocity_cpu(fields, momentum);
    // zero net streamwise flux (closed DHVC): subtract the bulk U the net
    // buoyancy would drive in the periodic streamwise direction.
    double net_dpdx = 0.0;
    physics::apply_mass_flow_correction_cpu(domain, fields, 0.0, dt, total_vol, net_dpdx);
    core::sync_field_cpu(fields, core::Var::U, bc, sub);
    core::sync_field_cpu(fields, core::Var::V, bc, sub);
    core::sync_field_cpu(fields, core::Var::W, bc, sub);
    zero_w_wall(fields[core::Var::W], sub);

    // (3) pressure: div → Poisson → project
    equation::solve_pressure_cpu(domain, bc, fields, pressure, dt);

    time += static_cast<double>(dt);
    ++step;

    // (4) statistics (after transient) + monitor
    if (step >= nstat_start && (step % nstat_int == 0))
      post::accumulate_statistics_cpu(stats, domain, fields);
    if (step % nmonitor == 0) {
      const double dv  = div_max(domain, fields);
      const double red = post::compute_re_delta_star_cpu(stats, nu, domain);
      if (mpi.is_root())
        std::printf("step=%d t=%.3g dt=%.3g div=%.2e Re_dstar=%.3f (target~%.3f)\n",
                    step, time, double(dt), dv, red, 0.23 * std::pow(Ra, 0.28));
    }
  }

  const double re_dstar = post::compute_re_delta_star_cpu(stats, nu, domain);
  post::write_statistics_cpu(stats, dir_stats + "stats_final.dat", step, nu, domain);
  if (mpi.is_root())
    std::printf("[dhvc] done: Ra=%.2e  Re_dstar=%.4f  (Ng et al. 0.23*Ra^0.28 = %.4f)\n",
                Ra, re_dstar, 0.23 * std::pow(Ra, 0.28));
  return 0;
}
