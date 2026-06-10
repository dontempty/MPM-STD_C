// apps/rbc — Rayleigh-Bénard convection (PaScaL_TCS Fig 9).
//
// Stage 1 = OB (constant properties): validates the RBC geometry + the central-
// temperature diagnostic before NOB (P7). Reuses the validated DHVC physics with
// the gravity axis rotated to the wall-normal:
//   x, y = horizontal (periodic, Lx = Ly = Γ·H)
//   z    = vertical / wall-normal / gravity (wall, Lz = H)
//          hot bottom θ=+0.5 at z=0, cold top θ=−0.5 at z=Lz; no-slip U,V,W.
// Buoyancy F = θ on W (z-momentum). Gravity is in the BOUNDED z, so there is no
// periodic net-flux runaway (unlike DHVC) → no zero-net-flux correction.
// OB free-fall nondim: nu = √(Pr/Ra), alpha = 1/√(Ra·Pr).
// Recipe each step: energy → momentum(+buoyancy on W) → pressure → project.
// Diagnostic: central temperature Θ_c = <θ>_{x,y,t} at z = Lz/2. OB ⇒ Θ_c ≈ 0;
// NOB (P7) ⇒ Θ_c > 0 (Fig 9). NOB properties/cross-stress land in P7.

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

// Global x,y-plane mean of θ at the centre z-plane (z = Lz/2). Θ_c diagnostic.
double center_temp(const core::Domain& d, const core::CpuFields& f) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const int nz_g = d.sub.n_global()[2], kc_g = nz_g / 2;
  const int kstart = d.sub.global_offset()[2], nz_loc = d.sub.n_interior()[2];
  const real_t* T = f[core::Var::T].data();
  double local = 0.0;
  if (kc_g >= kstart && kc_g < kstart + nz_loc) {
    const int kc = kc_g - kstart + h;
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j) local += double(T[(i * n2 + j) * n3 + kc]);
  }
  double g = 0.0;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_SUM, d.sub.topology().cart_comm());
  const int nx = d.sub.n_global()[0], ny = d.sub.n_global()[1];
  return g / (static_cast<double>(nx) * ny);
}

} // anonymous namespace

int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);

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

  core::CpuMomentumSystem momentum(sub);
  core::CpuScalarSystem   scalar_system(sub);
  core::PressureSystem    pressure;
  post::Stats             stats;
  post::init_statistics_cpu(stats, domain);

  // RBC: gravity along z (wall-normal, vertical), F = θ on W, coeff = 1 (OB).
  physics::BuoyancyParams buoy; buoy.gravity_axis = 2; buoy.coeff = 1; buoy.T_ref = 0;

  const double t_end    = cfg.get_or<double>("time", "t_end", 400.0);
  const real_t dt_cap   = static_cast<real_t>(cfg.get_or<double>("time", "dt_start", 0.02));
  const real_t max_cfl  = static_cast<real_t>(cfg.get_or<double>("time", "MaxCFL", 0.5));
  const int    n_steps  = cfg.get_or<int>("time", "n_steps", 200000);
  const int    nmonitor    = cfg.get_or<int>("output", "nmonitor", 200);
  const int    nstat_start = cfg.get_or<int>("output", "nstat_start", 20000);
  const std::string dir_stats = cfg.get_or<std::string>("output", "dir_stats", "apps/rbc/statistics/");
  if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
  MPI_Barrier(topo.cart_comm());

  // ── IC: linear conductive T(z) (hot bottom → cold top) + multi-mode
  //    perturbation to seed convection; fluid starts from rest (U=V=W=0). ─────
  {
    const double Lx = cfg.get<double>("domain", "Lx");
    const double Ly = cfg.get<double>("domain", "Ly");
    const double Lz = cfg.get<double>("domain", "Lz");
    const double T_lo = bc.T.face(Direction::Z, Side::Minus).value;   // hot (+0.5) at z=0
    const double T_hi = bc.T.face(Direction::Z, Side::Plus ).value;   // cold (−0.5) at z=Lz
    const double amp  = cfg.get_or<double>("init", "perturb", 0.02);
    const auto& xc = grid.xc(Direction::X); const auto& yc = grid.xc(Direction::Y); const auto& zc = grid.xc(Direction::Z);
    const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
    real_t* W = fields[core::Var::W].data(); real_t* T = fields[core::Var::T].data();
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        for (int k = 0; k < n3; ++k) {
          const int c = (i * n2 + j) * n3 + k;
          const double z = zc[k], x = xc[i], y = yc[j];
          T[c] = static_cast<real_t>(T_lo + (T_hi - T_lo) * (z / Lz));
          const double env = std::sin(M_PI * z / Lz);   // 0 at walls
          W[c] += static_cast<real_t>(amp * env * (std::sin(2.0 * M_PI * x / Lx) * std::cos(2.0 * M_PI * y / Ly)
                                                  + std::cos(2.0 * M_PI * x / Lx) * std::sin(2.0 * M_PI * y / Ly)
                                                  + std::sin(4.0 * M_PI * x / Lx) * std::sin(4.0 * M_PI * y / Ly)));
        }
  }
  zero_w_wall(fields[core::Var::W], sub);
  core::sync_field_cpu(fields, core::Var::U, bc, sub); core::sync_field_cpu(fields, core::Var::V, bc, sub);
  core::sync_field_cpu(fields, core::Var::W, bc, sub); core::sync_field_cpu(fields, core::Var::P, bc, sub);
  core::sync_field_cpu(fields, core::Var::T, bc, sub);

  if (mpi.is_root())
    std::printf("[rbc] OB  Ra=%.2e Pr=%.4g  nu=%.4e alpha=%.4e  n=%dx%dx%d (x,y periodic, z wall=gravity)\n",
                Ra, Pr, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

  // ── time loop ──────────────────────────────────────────────────────────────
  double tc_sum = 0.0; long tc_n = 0;   // running mean of central temperature
  double time = 0.0; int step = 0;
  while (time < t_end && step < n_steps) {
    const real_t dt = driver::compute_cfl_dt_cpu(domain, fields, max_cfl, dt_cap);

    equation::assemble_scalar_const_diff_cpu(domain, fields, scalar_system, dt);
    equation::solve_scalar_cpu(domain, bc, fields, scalar_system, dt);
    core::sync_field_cpu(fields, core::Var::T, bc, sub);

    equation::assemble_momentum_const_visc_cpu(domain, fields, momentum, dt);
    physics::add_buoyancy_force_cpu(momentum, fields[core::Var::T], buoy, dt);   // F=θ on W
    equation::solve_momentum_cpu(domain, bc, fields, momentum, dt);
    equation::update_velocity_cpu(fields, momentum);
    core::sync_field_cpu(fields, core::Var::U, bc, sub);
    core::sync_field_cpu(fields, core::Var::V, bc, sub);
    core::sync_field_cpu(fields, core::Var::W, bc, sub);
    zero_w_wall(fields[core::Var::W], sub);

    equation::solve_pressure_cpu(domain, bc, fields, pressure, dt);

    time += static_cast<double>(dt);
    ++step;

    if (step >= nstat_start) { tc_sum += center_temp(domain, fields); ++tc_n; }
    if (step % nmonitor == 0) {
      const double dv = div_max(domain, fields);
      const double tc = (tc_n > 0) ? tc_sum / static_cast<double>(tc_n) : center_temp(domain, fields);
      if (mpi.is_root())
        std::printf("step=%d t=%.3g dt=%.3g div=%.2e Theta_c=%.4f (OB->0)\n",
                    step, time, double(dt), dv, tc);
    }
  }

  const double theta_c = (tc_n > 0) ? tc_sum / static_cast<double>(tc_n) : center_temp(domain, fields);
  if (mpi.is_root())
    std::printf("[rbc] done: OB Ra=%.2e Pr=%.4g  Theta_c=%.5f  (OB reference -> 0)\n", Ra, Pr, theta_c);
  return 0;
}
