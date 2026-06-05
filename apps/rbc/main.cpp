// =============================================================================
//  apps/rbc/main.cpp  —  Rayleigh-Bénard / DHVC convection
//
//  Run with:
//      mpirun -np 1 build/cpu/bin/rbc apps/rbc/input.toml
// =============================================================================

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "config/main.hpp"
#include "equation/main.hpp"
#include "equation/pressure/pressure_solver_factory.hpp"
#include "field/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "parallel/main.hpp"
#include "post/main.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace mpmstd;


int main(int argc, char** argv) {

  // -------------------------------------------------------------------------
  //  (1)  MPI startup.
  // -------------------------------------------------------------------------
  parallel::mpi::MpiContext mpi(&argc, &argv);


  // -------------------------------------------------------------------------
  //  (2)  Logger + config file.
  // -------------------------------------------------------------------------
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);

  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);
  config::Logger::info("loaded config from '%s'", cfg.source_path().c_str());


  // -------------------------------------------------------------------------
  //  (3)  Cartesian MPI topology.
  // -------------------------------------------------------------------------
  std::array<int, 3> dims = {
    cfg.get<int>("mpi", "np1"),
    cfg.get<int>("mpi", "np2"),
    cfg.get<int>("mpi", "np3"),
  };
  auto axis_to_periodic = [](const std::string& s) -> bool {
    return s == "periodic";
  };
  std::array<bool, 3> periodic = {
    axis_to_periodic(cfg.get_or<std::string>("topology", "x", "periodic")),
    axis_to_periodic(cfg.get_or<std::string>("topology", "y", "periodic")),
    axis_to_periodic(cfg.get_or<std::string>("topology", "z", "periodic")),
  };
  parallel::mpi::MpiTopology topo(mpi, dims, periodic);


  // -------------------------------------------------------------------------
  //  (4)  Subdomain decomposition.
  // -------------------------------------------------------------------------
  std::array<int, 3> n_global = {
    cfg.get<int>("mesh", "n1m"),
    cfg.get<int>("mesh", "n2m"),
    cfg.get<int>("mesh", "n3m"),
  };
  parallel::mpi::Subdomain sub(topo, n_global);


  // -------------------------------------------------------------------------
  //  (5)  Grid (per-axis coordinates + metrics).
  // -------------------------------------------------------------------------
  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a) {
    const std::string gamma_key = "gamma"   + std::to_string(a + 1);
    const std::string uni_key   = "uniform" + std::to_string(a + 1);
    const std::string len_key   = std::string("L") + char('x' + a);

    const int    is_uniform = cfg.get_or<int>("uniform_mesh",  uni_key,   1);
    const double gamma      = cfg.get_or<double>("mesh_stretch", gamma_key, 0.0);
    const double L          = cfg.get<double>("domain", len_key);

    axes[a] = grid::AxisConfig{
      /*.n_global=*/ n_global[a],
      /*.length  =*/ L,
      /*.stretch =*/ is_uniform ? grid::StretchKind::Uniform
                                : grid::StretchKind::Tanh,
      /*.gamma   =*/ gamma,
    };
  }
  grid::Grid g(sub, axes);


  // -------------------------------------------------------------------------
  //  (6)  Problem  =  Topology + per-field BC table.
  // -------------------------------------------------------------------------
  boundary::Problem problem = boundary::load_problem_from_config(cfg);
  problem.validate();


  // -------------------------------------------------------------------------
  //  (7)  Backend.
  // -------------------------------------------------------------------------
  auto backend = parallel::make_default_backend();


  // -------------------------------------------------------------------------
  //  (8)  FieldRegistry.
  // -------------------------------------------------------------------------
  field::FieldRegistry fields(sub, *backend);
  auto& U  = fields.add_scalar("U");
  auto& V  = fields.add_scalar("V");
  auto& W  = fields.add_scalar("W");
  auto& dU = fields.add_scalar("dU");
  auto& dV = fields.add_scalar("dV");
  auto& dW = fields.add_scalar("dW");
  auto& P  = fields.add_scalar("P");
  auto& dP = fields.add_scalar("dP");
  auto& T  = fields.add_scalar("T");


  // -------------------------------------------------------------------------
  //  (9)  BoundaryApplier.
  // -------------------------------------------------------------------------
  boundary::BoundaryApplier bc(problem);


  // -------------------------------------------------------------------------
  //  (10) TdmaRegistry.
  // -------------------------------------------------------------------------
  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);


  // -------------------------------------------------------------------------
  //  (11) Physics constants.
  // -------------------------------------------------------------------------
  const double Ra      = cfg.get<double>("physics", "Ra");
  const double Pr      = cfg.get<double>("physics", "Pr");
  const double nu      = std::sqrt(Pr  / Ra);
  const double alpha_T = std::sqrt(1.0 / (Ra * Pr));


  // -------------------------------------------------------------------------
  //  (12) Initial conditions.
  //
  //  [init] u_ic_dhvc = 1: start from analytical DHVC steady state.
  //  Bypasses the O(n²) CN transient from U=0 for spatial EOC studies.
  // -------------------------------------------------------------------------
  U.fill_host(0.0);   V.fill_host(0.0);   W.fill_host(0.0);
  dU.fill_host(0.0);  dV.fill_host(0.0);  dW.fill_host(0.0);
  P.fill_host(0.0);
  dP.fill_host(0.0);

  if (cfg.get_or<int>("init", "u_ic_dhvc", 0)) {
    const double Lz_ = cfg.get<double>("domain", "Lz");
    const auto&  zc_ = g.xc(Direction::Z);
    const int n1_ = sub.n_total()[0];
    const int n2_ = sub.n_total()[1];
    const int n3_ = sub.n_total()[2];
    for (int i = 0; i < n1_; ++i)
      for (int j = 0; j < n2_; ++j)
        for (int k = 0; k < n3_; ++k) {
          const double z_ = zc_[k];
          U.host_at(i, j, k) = static_cast<real_t>(
              z_ * (2.0*z_/Lz_ - 1.0) * (z_/Lz_ - 1.0) / (12.0 * nu));
        }
  }

  // Linear conductive temperature profile.
  {
    const double Lz      = cfg.get<double>("domain", "Lz");
    const double T_minus = problem.T.face(Direction::Z, Side::Minus).value(0, 0, 0, 0);
    const double T_plus  = problem.T.face(Direction::Z, Side::Plus ).value(0, 0, 0, 0);
    const auto&  zc      = g.xc(Direction::Z);

    const int n1 = sub.n_total()[0];
    const int n2 = sub.n_total()[1];
    const int n3 = sub.n_total()[2];
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        for (int k = 0; k < n3; ++k) {
          const double frac = zc[k] / Lz;
          T.host_at(i, j, k) = T_minus + (T_plus - T_minus) * frac;
        }
  }


  // -------------------------------------------------------------------------
  //  (13) Halo exchange + boundary fill.
  // -------------------------------------------------------------------------
  U.exchange_halo();  bc.apply_ghost(U, problem.U);
  V.exchange_halo();  bc.apply_ghost(V, problem.V);
  W.exchange_halo();  bc.apply_ghost(W, problem.W);
  P.exchange_halo();  bc.apply_ghost(P, problem.P);
  T.exchange_halo();  bc.apply_ghost(T, problem.T);


  // -------------------------------------------------------------------------
  //  Setup summary (root rank only).
  // -------------------------------------------------------------------------
  if (mpi.is_root()) {
    auto wax = problem.topology.wall_axis();
    auto so  = problem.topology.sweep_order();

    std::cout << "\n=== MPM-STD RBC setup complete ===\n"
              << "  ranks      : " << mpi.world_size() << "\n"
              << "  proc grid  : " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n"
              << "  mesh (n_m) : " << n_global[0] << " x " << n_global[1]
                                   << " x " << n_global[2] << "\n"
              << "  domain L   : " << cfg.get<double>("domain", "Lx") << " x "
                                   << cfg.get<double>("domain", "Ly") << " x "
                                   << cfg.get<double>("domain", "Lz") << "\n"
              << "  Ra / Pr    : " << Ra << " / " << Pr << "\n"
              << "  nu / alpha : " << nu << " / " << alpha_T << "\n"
              << "  T walls    : "
              << problem.T.face(Direction::Z, Side::Minus).value(0,0,0,0) << " (-z) , "
              << problem.T.face(Direction::Z, Side::Plus ).value(0,0,0,0) << " (+z)\n"
              << "  U.kind ±z  : "
              << boundary::bc_kind_name(problem.U.face(Direction::Z, Side::Minus).kind) << " / "
              << boundary::bc_kind_name(problem.U.face(Direction::Z, Side::Plus ).kind) << "\n"
              << "  wall axis  : "
              << (wax ? direction_name(wax.value()) : "(ambiguous)") << "\n"
              << "  sweep ord  : "
              << direction_name(so[0]) << " -> "
              << direction_name(so[1]) << " -> "
              << direction_name(so[2]) << "\n\n";
  }


  // -------------------------------------------------------------------------
  //  TIME LOOP
  //
  //  DHVC free-fall scaling: ν = sqrt(Pr/Ra), α_T = 1/sqrt(Ra·Pr)
  //  Boussinesq buoyancy on U:  f_x = T
  // -------------------------------------------------------------------------
  const double t_start = cfg.get_or<double>("time", "t_start",  0.0);
  const double t_end   = cfg.get<double>("time", "t_end");
  const double MaxCFL  = cfg.get_or<double>("time", "MaxCFL",   1.0);
  const int    n_steps = cfg.get_or<int>   ("time", "n_steps",  std::numeric_limits<int>::max());

  // --- Thermal equation ------------------------------------------------
  equation::scalar::ScalarTraits T_traits;
  T_traits.name            = "T";
  T_traits.diffusivity     = static_cast<real_t>(alpha_T);
  T_traits.with_convection = true;

  equation::scalar::ScalarEquation thermal_eq(
      T_traits, g, sub, fields, problem, problem.T, *tdma, bc);

  // --- Momentum equations (U, V, W) ------------------------------------
  equation::momentum::MomentumTraits U_traits, V_traits, W_traits;
  U_traits.name            = "U";
  U_traits.viscosity       = static_cast<real_t>(nu);
  U_traits.with_convection = true;
  U_traits.source_name     = "T";   // Boussinesq buoyancy: f_x = T

  V_traits.name            = "V";
  V_traits.viscosity       = static_cast<real_t>(nu);
  V_traits.with_convection = true;

  W_traits.name            = "W";
  W_traits.viscosity       = static_cast<real_t>(nu);
  W_traits.with_convection = true;

  equation::momentum::MomentumEquation mom_U(
      U_traits, g, sub, fields, problem, problem.U, *tdma, bc);
  equation::momentum::MomentumEquation mom_V(
      V_traits, g, sub, fields, problem, problem.V, *tdma, bc);
  equation::momentum::MomentumEquation mom_W(
      W_traits, g, sub, fields, problem, problem.W, *tdma, bc);

  // --- Pressure solver -------------------------------------------------
  auto pressure_solver = equation::pressure::make_pressure_solver(
      g, sub, fields, problem, *tdma, bc);

  // --- CFL helper ------------------------------------------------------
  auto compute_max_cfl = [&](real_t dt) -> double {
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0];
    const int n2 = sub.n_total()[1];
    const int n3 = sub.n_total()[2];

    const real_t* u   = fields.scalar("U").host_ptr();
    const real_t* v   = fields.scalar("V").host_ptr();
    const real_t* w   = fields.scalar("W").host_ptr();
    const real_t* dx1 = g.dx_ptr(Direction::X);
    const real_t* dx2 = g.dx_ptr(Direction::Y);
    const real_t* dx3 = g.dx_ptr(Direction::Z);

    double local_max = 1e-20;
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j)
        for (int k = h; k < n3 - h; ++k) {
          const int p = (i * n2 + j) * n3 + k;
          const double c = (std::abs(static_cast<double>(u[p])) / dx1[i]
                          + std::abs(static_cast<double>(v[p])) / dx2[j]
                          + std::abs(static_cast<double>(w[p])) / dx3[k]);
          if (c > local_max) local_max = c;
        }

    double global_max = local_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  topo.cart_comm());
    return global_max * static_cast<double>(dt);
  };

  // --- Divergence check ------------------------------------------------
  auto max_divergence = [&]() -> double {
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0];
    const int n2 = sub.n_total()[1];
    const int n3 = sub.n_total()[2];

    const real_t* u   = fields.scalar("U").host_ptr();
    const real_t* v   = fields.scalar("V").host_ptr();
    const real_t* w   = fields.scalar("W").host_ptr();
    const real_t* dx1 = g.dx_ptr(Direction::X);
    const real_t* dx2 = g.dx_ptr(Direction::Y);
    const real_t* dx3 = g.dx_ptr(Direction::Z);

    double local_max = 0.0;
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j)
        for (int k = h; k < n3 - h; ++k) {
          const int p   = (i * n2 + j) * n3 + k;
          const int pip = ((i+1)*n2 + j) * n3 + k;
          const int pjp = (i * n2 + j+1) * n3 + k;
          const int pkp = (i * n2 + j) * n3 + k+1;
          const double div = (u[pip] - u[p]) / dx1[i]
                           + (v[pjp] - v[p]) / dx2[j]
                           + (w[pkp] - w[p]) / dx3[k];
          const double adiv = std::abs(div);
          if (adiv > local_max) local_max = adiv;
        }

    double global_max = local_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  topo.cart_comm());
    return global_max;
  };

  // --- Time loop -------------------------------------------------------
  real_t dt   = static_cast<real_t>(cfg.get_or<double>("time", "dt_start", 0.05));
  double time = t_start;
  int    step = 0;

  if (mpi.is_root()) {
    std::cout << "=== time loop start  t_end=" << t_end
              << "  n_steps=" << n_steps << " ===\n" << std::flush;
  }

  while (time < t_end && step < n_steps) {

    // 1. Thermal: T^n → T^{n+1}
    thermal_eq.step(dt);

    // 2. Momentum predictor
    mom_U.step(dt);
    mom_V.step(dt);
    mom_W.step(dt);

    // W wall face at interior index k=h: enforce no-penetration explicitly.
    {
      const int n1t = sub.n_total()[0], n2t = sub.n_total()[1], n3t = sub.n_total()[2];
      const int h   = kHaloWidth;
      real_t* w = fields.scalar("W").host_ptr();
      const bool lower_wall = (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
      const bool upper_wall = (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
      if (lower_wall)
        for (int i = 0; i < n1t; ++i)
          for (int j = 0; j < n2t; ++j)
            w[(i * n2t + j) * n3t + h] = real_t{0};
      if (upper_wall)
        for (int i = 0; i < n1t; ++i)
          for (int j = 0; j < n2t; ++j)
            w[(i * n2t + j) * n3t + (n3t - h)] = real_t{0};
    }

    // 3. Pressure + projection
    pressure_solver->solve(dt, fields.scalar("U"), fields.scalar("V"),
                                fields.scalar("W"), fields.scalar("P"));

    time += static_cast<double>(dt);
    ++step;

    // 4. Diagnostics
    if (mpi.is_root()) {
      const double div = max_divergence();
      const double cfl = compute_max_cfl(dt);
      std::cout << "  step " << step
                << "  t=" << time
                << "  dt=" << static_cast<double>(dt)
                << "  CFL=" << cfl
                << "  div=" << div
                << "\n" << std::flush;
    }

    // 5. CFL-based dt update (20% growth cap)
    {
      const double speed_sum = compute_max_cfl(real_t{1.0});
      if (speed_sum > 1e-14) {
        const real_t dt_cfl = static_cast<real_t>(MaxCFL / speed_sum);
        dt = std::min(dt_cfl, static_cast<real_t>(1.2) * dt);
      }
    }
  }

  if (mpi.is_root()) {
    std::cout << "\n=== time loop complete: " << step << " steps, t=" << time << " ===\n\n";
  }

  // Binary dump for PaScaL_TCS / EOC comparison.
  // Layout: outer=stream(X), middle=wall(Z), inner=span(Y)
  if (mpi.is_root()) {
    const int h   = kHaloWidth;
    const int n2t = sub.n_total()[1];
    const int n3t = sub.n_total()[2];
    const int n1m = sub.n_interior()[0];
    const int n2m = sub.n_interior()[1];
    const int n3m = sub.n_interior()[2];

    const real_t* t = fields.scalar("T").host_ptr();
    FILE* f = std::fopen("dump_mpm_T.bin", "wb");
    for (int ii = 0; ii < n1m; ++ii)
      for (int kk = 0; kk < n3m; ++kk)
        for (int jj = 0; jj < n2m; ++jj) {
          double v = static_cast<double>(
              t[(ii + h) * n2t * n3t + (jj + h) * n3t + (kk + h)]);
          std::fwrite(&v, sizeof(double), 1, f);
        }
    std::fclose(f);
    std::cout << "Wrote dump_mpm_T.bin  (" << n1m << "x" << n3m << "x" << n2m << " doubles)\n";

    const real_t* u = fields.scalar("U").host_ptr();
    FILE* fu = std::fopen("dump_mpm_U.bin", "wb");
    for (int ii = 0; ii < n1m; ++ii)
      for (int kk = 0; kk < n3m; ++kk)
        for (int jj = 0; jj < n2m; ++jj) {
          double v = static_cast<double>(
              u[(ii + h) * n2t * n3t + (jj + h) * n3t + (kk + h)]);
          std::fwrite(&v, sizeof(double), 1, fu);
        }
    std::fclose(fu);
    std::cout << "Wrote dump_mpm_U.bin  (" << n1m << "x" << n3m << "x" << n2m << " doubles)\n";
  }

  return 0;
}
