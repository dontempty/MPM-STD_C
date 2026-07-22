// apps/validate — PaScaL_TCS momentum/energy validation harness.
//
// Reproduces the reference ground-truth case (tests/regression/ground_truth/
// CASE_SPEC.md): DHVC geometry with gravity along the STREAM axis (my X),
// walls on Z (Dirichlet T ±0.5), span Y; deterministic IC; FIXED dt; 200 steps;
// OB (props≡1). Dumps U,V,W,T at the final step in the reference layout
// (i=X outer, j=Z middle, k=Y inner) for compare_pascal_tcs.py.
//
//   physics.faithful = 1 → faithful variable-μ momentum (the new port).
//   physics.faithful = 0 → current simple const-visc momentum (baseline gap).
// Both share the same faithful energy + the reference face-interpolated buoyancy
// on U, so the ONLY variable is the momentum scheme. Serial (np = 1,1,1).

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
#include "physics/properties/properties.hpp"
#include "driver/cfl.hpp"

#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

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

// Reference face-interpolated buoyancy on U (ref solvedU 347-352):
//   rhs_u += dt · invRhoc · (Tc + a12pera11·Tc²·ΔT),  Cmt=1,
//   Tc = 0.5·(dx1(im)·T(i) + dx1(i)·T(im))/dmx1(i) ,  invRhoc face-interp.
void add_buoyancy_face_u(core::CpuMomentumSystem& mom, const core::CpuField& T,
                         const core::CpuField& inv_rho, const core::Domain& d,
                         real_t a12pera11, real_t dT, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* t = T.data(); const real_t* ir = inv_rho.data();
  const real_t* dx1 = d.grid.dx_ptr(Direction::X), *dmx1 = d.grid.dmx_ptr(Direction::X);
  real_t* rhs = mom.rhs_u.data();
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k, cim = ((i - 1) * n2 + j) * n3 + k;
        const real_t Tc  = real_t{0.5} * (dx1[i - 1] * t[c]  + dx1[i] * t[cim])  / dmx1[i];
        const real_t irc = real_t{0.5} * (dx1[i - 1] * ir[c] + dx1[i] * ir[cim]) / dmx1[i];
        rhs[c] += dt * irc * (Tc + a12pera11 * Tc * Tc * dT);
      }
}

// Dump interior field in the reference layout: i=X outer, j=Z middle, k=Y inner.
// my storage idx = (iX*n2 + jY)*n3 + kZ.  Serial → interior = full domain.
void dump_field(const core::CpuField& F, const core::Subdomain& sub, const std::string& path) {
  const auto nt = sub.n_total(); const int n2 = nt[1], n3 = nt[2];
  const auto ni = sub.n_interior(); const int i1 = ni[0], i2 = ni[1], i3 = ni[2];
  const int h = kHaloWidth;
  const real_t* f = F.data();
  std::vector<double> buf;
  buf.reserve(static_cast<size_t>(i1) * i2 * i3);
  for (int ii = 0; ii < i1; ++ii)             // X outer
    for (int kk = 0; kk < i3; ++kk)           // Z middle
      for (int jj = 0; jj < i2; ++jj)         // Y inner
        buf.push_back(double(f[((ii + h) * n2 + (jj + h)) * n3 + (kk + h)]));
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  std::fwrite(buf.data(), sizeof(double), buf.size(), fp);
  std::fclose(fp);
}

} // anonymous namespace

int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/validate/input.toml";
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

  const bool   faithful = cfg.get_or<int>("physics", "faithful", 1) != 0;
  const double DeltaT   = cfg.get_or<double>("physics", "DeltaT", 0.0);
  physics::CpuPropertyFields   props(sub);
  const physics::PropertyModel pmodel = physics::PropertyModel::glycerol(static_cast<real_t>(DeltaT));
  const real_t a12pera11 = (std::fabs(pmodel.a[2]) < static_cast<real_t>(1e-14)) ? 0 : pmodel.a[2] / pmodel.a[1];

  const real_t dt_fixed = static_cast<real_t>(cfg.get_or<double>("time", "dt_start", 0.004));
  const real_t max_cfl  = static_cast<real_t>(cfg.get_or<double>("time", "MaxCFL", 1e8));
  const int    n_steps  = cfg.get_or<int>("time", "n_steps", 200);
  const int    nmonitor = cfg.get_or<int>("output", "nmonitor", 50);
  const std::string out_prefix = cfg.get_or<std::string>("output", "dump_prefix", "dump_mpm");

  // ── IC: T = T_lo + (T_hi−T_lo)·z/Lz + amp·sin(2πx/Lx)·sin(πz/Lz)·cos(2πy/Ly);
  //    velocity from rest (U=V=W=0).  (CASE_SPEC; axis: x1→X, x2/wall→Z, x3→Y) ─
  {
    const double Lx = cfg.get<double>("domain", "Lx");
    const double Ly = cfg.get<double>("domain", "Ly");
    const double Lz = cfg.get<double>("domain", "Lz");
    const double T_lo = bc.T.face(Direction::Z, Side::Minus).value;   // hot (+0.5) at z=0
    const double T_hi = bc.T.face(Direction::Z, Side::Plus ).value;   // cold (−0.5) at z=Lz
    const double amp  = cfg.get_or<double>("init", "perturb", 0.05);
    const auto& xc = grid.xc(Direction::X); const auto& yc = grid.xc(Direction::Y); const auto& zc = grid.xc(Direction::Z);
    const auto nt = sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
    real_t* T = fields[core::Var::T].data();
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        for (int k = 0; k < n3; ++k) {
          const int c = (i * n2 + j) * n3 + k;
          const double x = xc[i], y = yc[j], z = zc[k];
          T[c] = static_cast<real_t>(T_lo + (T_hi - T_lo) * (z / Lz)
                 + amp * std::sin(2.0 * M_PI * x / Lx) * std::sin(M_PI * z / Lz) * std::cos(2.0 * M_PI * y / Ly));
        }
  }
  core::sync_field_cpu(fields, core::Var::U, bc, sub); core::sync_field_cpu(fields, core::Var::V, bc, sub);
  core::sync_field_cpu(fields, core::Var::W, bc, sub); core::sync_field_cpu(fields, core::Var::P, bc, sub);
  core::sync_field_cpu(fields, core::Var::T, bc, sub);

  if (mpi.is_root())
    std::printf("[validate] %s momentum  Ra=%.1f Pr=%.3g  nu=%.6f  n=%dx%dx%d  dt=%.4f x%d steps\n",
                faithful ? "FAITHFUL(var)" : "CURRENT(const)", Ra, Pr, nu,
                n_global[0], n_global[1], n_global[2], double(dt_fixed), n_steps);

  // ── time loop (CASE_SPEC order: energy → momentum(+buoyancy U) → project) ────
  double time = 0.0; int step = 0;
  while (step < n_steps) {
    const real_t dt = std::min(dt_fixed, driver::compute_cfl_dt_cpu(domain, fields, max_cfl, dt_fixed));

    physics::update_properties_cpu(props, fields[core::Var::T], pmodel);   // props≡1 at ΔT=0

    // energy: always the faithful var path (committed) so only momentum varies.
    equation::solve_scalar_nob_cpu(domain, bc, fields, props.KPP, props.dKPP,
                                   props.invRhoCp, props.dinvRhoCp, scalar_system, dt);
    core::sync_field_cpu(fields, core::Var::T, bc, sub);

    if (faithful) equation::assemble_momentum_var_visc_cpu(domain, fields, props.Mu, props.invRho, momentum, dt);
    else          equation::assemble_momentum_const_visc_cpu(domain, fields, momentum, dt);
    add_buoyancy_face_u(momentum, fields[core::Var::T], props.invRho, domain, a12pera11, static_cast<real_t>(DeltaT), dt);
    if (faithful) equation::solve_momentum_var_cpu(domain, bc, fields, props.Mu, props.invRho, momentum, dt);
    else          equation::solve_momentum_cpu(domain, bc, fields, momentum, dt);
    equation::update_velocity_cpu(fields, momentum);
    core::sync_field_cpu(fields, core::Var::U, bc, sub);
    core::sync_field_cpu(fields, core::Var::V, bc, sub);
    core::sync_field_cpu(fields, core::Var::W, bc, sub);
    zero_w_wall(fields[core::Var::W], sub);

    equation::solve_pressure_cpu(domain, bc, fields, pressure, dt);

    time += static_cast<double>(dt);
    ++step;
    if (step % nmonitor == 0 && mpi.is_root())
      std::printf("step=%d t=%.4f dt=%.4f div=%.2e\n", step, time, double(dt), div_max(domain, fields));
  }

  // ── dump in reference layout, named by the REFERENCE field they match (so the
  //    comparison is 1:1 by name): my U→_U, my W→_V(wall), my V→_W(span), my T→_T.
  dump_field(fields[core::Var::U], sub, out_prefix + "_U.bin");   // ref U  (x1 stream) ↔ my X
  dump_field(fields[core::Var::W], sub, out_prefix + "_V.bin");   // ref V  (x2 wall)   ↔ my Z
  dump_field(fields[core::Var::V], sub, out_prefix + "_W.bin");   // ref W  (x3 span)   ↔ my Y
  dump_field(fields[core::Var::T], sub, out_prefix + "_T.bin");
  if (mpi.is_root())
    std::printf("[validate] done t=%.4f  dumped %s_{U,V,W,T}.bin  (div=%.2e)\n",
                time, out_prefix.c_str(), div_max(domain, fields));
  return 0;
}
