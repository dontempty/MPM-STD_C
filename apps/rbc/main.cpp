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
#include "equation/momentum/linear_system.hpp"
#include "equation/scalar/linear_system.hpp"
#include "equation/pressure/linear_system.hpp"
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
#include "physics/properties/properties.hpp"   // NOB glycerol properties (var-μ path)
#include "driver/cfl.hpp"
#include "post/statistics.hpp"

#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace mpmstd;

namespace {

void zero_w_wall(core::CpuField& W, const parallel::mpi::Subdomain& sub) {
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

// Diagnostic: global max|U|,max|V|,max|W| and T range (+ wall-normal k of max|W|).
// Distinguishes a momentum/buoyancy velocity spike from a scalar (T) overshoot,
// and whether the velocity max sits at the wall (small/large k) or the interior.
struct FlowDiag { double umax, vmax, wmax, tmax, tmin; int kwmax; };
FlowDiag flow_diag(const core::Domain& d, const core::CpuFields& f) {
  const auto nt = d.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* u = f[core::Var::U].data(); const real_t* v = f[core::Var::V].data();
  const real_t* w = f[core::Var::W].data(); const real_t* T = f[core::Var::T].data();
  double um=0, vm=0, wm=0, tx=-1e30, tn=1e30; int kw=-1; double wbest=-1;
  const int kstart = d.sub.global_offset()[2];
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
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
  parallel::mpi::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);

  auto is_per = [](const std::string& s) { return s == "periodic"; };
  std::array<int, 3>  dims     = {cfg.get<int>("mpi", "np1"), cfg.get<int>("mpi", "np2"), cfg.get<int>("mpi", "np3")};
  std::array<bool, 3> periodic = {is_per(cfg.get_or<std::string>("topology", "x", "periodic")),
                                  is_per(cfg.get_or<std::string>("topology", "y", "periodic")),
                                  is_per(cfg.get_or<std::string>("topology", "z", "wall"))};
  parallel::mpi::MpiTopology topo(mpi, dims, periodic);
  std::array<int, 3> n_global = {cfg.get<int>("mesh", "n1m"), cfg.get<int>("mesh", "n2m"), cfg.get<int>("mesh", "n3m")};
  parallel::mpi::Subdomain sub(topo, n_global);

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a) {
    const int    is_uni = cfg.get_or<int>("uniform_mesh", "uniform" + std::to_string(a + 1), 1);
    const double gamma  = cfg.get_or<double>("mesh_stretch", "gamma" + std::to_string(a + 1), 0.0);
    const double L      = cfg.get<double>("domain", std::string("L") + char('x' + a));
    axes[a] = {n_global[a], L, is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
  }
  grid::Grid grid(sub, axes);
  boundary::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
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

  equation::momentum::CpuLinearSystem momentum(sub);
  equation::scalar::CpuLinearSystem   scalar_system(sub);
  equation::pressure::LinearSystem    pressure;
  post::Stats             stats;
  post::init_statistics_cpu(stats, domain);

  // RBC: gravity along z (wall-normal, vertical), F = θ on W, coeff = 1 (OB).
  physics::buoyancy::BuoyancyParams buoy; buoy.gravity_axis = 2; buoy.coeff = 1; buoy.T_ref = 0;

  // NOB (variable-property) path: nob=1 → variable-μ momentum driven by the
  // glycerol property fields. DeltaT sets the polynomial nonlinearity; DeltaT=0
  // ⇒ properties≡1, so nob=1/DeltaT=0 validates the var path against const-visc.
  const bool   nob    = cfg.get_or<int>("physics", "nob", 0) != 0;
  const double DeltaT = cfg.get_or<double>("physics", "DeltaT", 0.0);
  physics::properties::CpuPropertyFields  props(sub);
  const physics::properties::PropertyModel pmodel = physics::properties::PropertyModel::glycerol(static_cast<real_t>(DeltaT));
  // NOB buoyancy nonlinearity: a12pera11 = a12/a11 of the density polynomial
  // (NOB_RBC_SPEC.md §5); guarded so OB (a12=0) and DeltaT=0 both give F = θ.
  buoy.a12pera11 = (std::fabs(pmodel.a[2]) < static_cast<real_t>(1e-14)) ? 0 : pmodel.a[2] / pmodel.a[1];
  buoy.delta_T   = static_cast<real_t>(DeltaT);

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

#ifdef MPMSTD_BACKEND_CUDA
  // G3b: GPU hybrid — scalar+momentum on GPU, pressure on CPU.
  core::bind_gpu_to_local_rank_gpu(mpi);

  core::GpuFields gpu_fields;
  gpu_fields.add(core::Var::U, sub); gpu_fields.add(core::Var::V, sub);
  gpu_fields.add(core::Var::W, sub); gpu_fields.add(core::Var::P, sub);
  gpu_fields.add(core::Var::T, sub);
  gpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  gpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  equation::momentum::GpuLinearSystem gpu_momentum(sub);
  equation::scalar::GpuLinearSystem   gpu_scalar(sub);
  physics::properties::GpuPropertyFields gpu_props(sub);

  gpu_fields[core::Var::U].to_device(fields[core::Var::U].data());
  gpu_fields[core::Var::V].to_device(fields[core::Var::V].data());
  gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
  gpu_fields[core::Var::P].to_device(fields[core::Var::P].data());
  gpu_fields[core::Var::T].to_device(fields[core::Var::T].data());
#endif

  // ── time loop ──────────────────────────────────────────────────────────────
  double tc_sum = 0.0; long tc_n = 0;   // running mean of central temperature
  double time = 0.0; int step = 0;

  // Nu + z-profile accumulators
  const int nzg  = sub.n_global()[2];
  const int koff = sub.global_offset()[2];
  const int nzl  = sub.n_interior()[2];
  std::vector<double> t_sum_z(nzg, 0.0);
  std::vector<double> w2_sum_z(nzg, 0.0);
  double wt_sum = 0.0;
  const real_t* dx3_z = grid.dx_ptr(Direction::Z);
  while (time < t_end && step < n_steps) {
#ifdef MPMSTD_BACKEND_CUDA
    // ── GPU hybrid path (G3b): scalar+momentum GPU, pressure CPU ────────────
    const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);

    if (nob) physics::properties::update_properties_gpu(gpu_props, gpu_fields[core::Var::T], pmodel);

    if (nob) {
      equation::scalar::solve_scalar_nob_gpu(domain, bc, gpu_fields, gpu_props.KPP, gpu_props.dKPP,
                                     gpu_props.invRhoCp, gpu_props.dinvRhoCp, gpu_scalar, dt);
    } else {
      equation::scalar::solve_scalar_ob_gpu(domain, bc, gpu_fields, gpu_scalar, dt);
    }
    core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);

    if (nob) equation::momentum::assemble_momentum_var_visc_gpu(domain, gpu_fields, gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
    else     equation::momentum::assemble_momentum_const_visc_gpu(domain, gpu_fields, gpu_momentum, dt);
    if (nob) physics::buoyancy::add_buoyancy_force_nob_gpu(domain, gpu_momentum, gpu_fields[core::Var::T], gpu_props.invRho, buoy, dt);
    else     physics::buoyancy::add_buoyancy_force_gpu(domain, gpu_momentum, gpu_fields[core::Var::T], buoy, dt);
    if (nob) equation::momentum::solve_momentum_var_gpu(domain, bc, gpu_fields, gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
    else     equation::momentum::solve_momentum_gpu(domain, bc, gpu_fields, gpu_momentum, dt);
    equation::momentum::update_velocity_gpu(gpu_fields, gpu_momentum);
    core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);

    // zero_w_wall workaround: transfer W, apply on host, transfer back
    gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
    zero_w_wall(fields[core::Var::W], sub);
    gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);

    // Transfer GPU→CPU for pressure and diagnostics
    gpu_fields[core::Var::U].to_host(fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_host(fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
    gpu_fields[core::Var::T].to_host(fields[core::Var::T].data());

    // Update CPU properties for pressure (from transferred T)
    if (nob) physics::properties::update_properties_cpu(props, fields[core::Var::T], pmodel);

    equation::pressure::solve_pressure_cpu(domain, bc, fields, pressure, dt,
                                 nob ? props.invRho.data() : nullptr);

    if (nob && DeltaT > 0.0) {
      real_t* t_ptr = fields[core::Var::T].data();
      const std::size_t nf = (std::size_t)sub.n_total()[0]*sub.n_total()[1]*sub.n_total()[2];
      for (std::size_t q = 0; q < nf; ++q)
        t_ptr[q] = std::max(real_t{-0.5}, std::min(real_t{0.5}, t_ptr[q]));
    }

    // Transfer back to GPU after pressure projection
    gpu_fields[core::Var::U].to_device(fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_device(fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
    gpu_fields[core::Var::T].to_device(fields[core::Var::T].data());

#else
    // ── CPU path ─────────────────────────────────────────────────────────────
    const real_t dt = driver::compute_cfl_dt_cpu(domain, fields, max_cfl, dt_cap);

    if (nob) physics::properties::update_properties_cpu(props, fields[core::Var::T], pmodel);

    if (nob) {
      equation::scalar::solve_scalar_nob_cpu(domain, bc, fields, props.KPP, props.dKPP, props.invRhoCp, props.dinvRhoCp, scalar_system, dt);
    } else {
      equation::scalar::coef_scalar_ob_cpu(domain, fields, scalar_system, dt);
      equation::scalar::solve_scalar_ob_cpu(domain, bc, fields, scalar_system, dt);
    }
    core::sync_field_cpu(fields, core::Var::T, bc, sub);

    if (nob) equation::momentum::assemble_momentum_var_visc_cpu(domain, fields, props.Mu, props.invRho, momentum, dt);
    else     equation::momentum::assemble_momentum_const_visc_cpu(domain, fields, momentum, dt);
    if (nob) physics::buoyancy::add_buoyancy_force_nob_cpu(domain, momentum, fields[core::Var::T], props.invRho, buoy, dt);
    else     physics::buoyancy::add_buoyancy_force_cpu(domain, momentum, fields[core::Var::T], buoy, dt);
    if (nob) equation::momentum::solve_momentum_var_cpu(domain, bc, fields, props.Mu, props.invRho, momentum, dt);
    else     equation::momentum::solve_momentum_cpu(domain, bc, fields, momentum, dt);
    equation::momentum::update_velocity_cpu(fields, momentum);
    core::sync_field_cpu(fields, core::Var::U, bc, sub);
    core::sync_field_cpu(fields, core::Var::V, bc, sub);
    core::sync_field_cpu(fields, core::Var::W, bc, sub);
    zero_w_wall(fields[core::Var::W], sub);

    equation::pressure::solve_pressure_cpu(domain, bc, fields, pressure, dt,
                                 nob ? props.invRho.data() : nullptr);

    if (nob && DeltaT > 0.0) {
      real_t* t_ptr = fields[core::Var::T].data();
      const std::size_t nf = (std::size_t)sub.n_total()[0]*sub.n_total()[1]*sub.n_total()[2];
      for (std::size_t q = 0; q < nf; ++q)
        t_ptr[q] = std::max(real_t{-0.5}, std::min(real_t{0.5}, t_ptr[q]));
    }
#endif

    time += static_cast<double>(dt);
    ++step;

    if (step >= nstat_start) {
      tc_sum += center_temp(domain, fields);
      ++tc_n;
      {
        const auto nt_ = sub.n_total(); const int n1_=nt_[0], n2_=nt_[1], n3_=nt_[2];
        const int h_ = kHaloWidth;
        const real_t* Tp = fields[core::Var::T].data();
        const real_t* Wp = fields[core::Var::W].data();
        double wt_loc = 0.0;
        for (int i = h_; i < n1_-h_; ++i)
          for (int j = h_; j < n2_-h_; ++j)
            for (int kk = 0; kk < nzl; ++kk) {
              const int k = kk + h_;
              const double tv = double(Tp[(i*n2_+j)*n3_+k]);
              const double wv = double(Wp[(i*n2_+j)*n3_+k]);
              t_sum_z[koff+kk] += tv;
              w2_sum_z[koff+kk] += wv*wv;
              wt_loc += wv * tv * double(dx3_z[k]);
            }
        double wt_g = 0.0;
        MPI_Allreduce(&wt_loc, &wt_g, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
        wt_sum += wt_g;
      }
    }
    if (step % nmonitor == 0) {
      const double dv = div_max(domain, fields);
      const double tc = (tc_n > 0) ? tc_sum / static_cast<double>(tc_n) : center_temp(domain, fields);
      const FlowDiag fd = flow_diag(domain, fields);
      if (mpi.is_root()) {
        std::printf("step=%d t=%.3g dt=%.3g div=%.2e Theta_c=%.4f | umax=%.3g vmax=%.3g wmax=%.3g(k=%d) T=[%.3f,%.3f]\n",
                    step, time, double(dt), dv, tc, fd.umax, fd.vmax, fd.wmax, fd.kwmax, fd.tmin, fd.tmax);
        if (tc_n > 0) {
          const int nxg_ = sub.n_global()[0], nyg_ = sub.n_global()[1];
          const double wt_now = wt_sum / (static_cast<double>(nxg_)*nyg_) / static_cast<double>(tc_n);
          std::printf("  [Nu] running: Nu=%.4f (tc_n=%ld)\n", 1.0 + wt_now/alpha_T, tc_n);
        }
      }
    }
  }

  const double theta_c = (tc_n > 0) ? tc_sum / static_cast<double>(tc_n) : center_temp(domain, fields);
  if (mpi.is_root())
    std::printf("[rbc] done: OB Ra=%.2e Pr=%.4g  Theta_c=%.5f  (OB reference -> 0)\n", Ra, Pr, theta_c);

  if (tc_n > 0) {
    const int nxg = sub.n_global()[0], nyg = sub.n_global()[1];
    std::vector<double> t_z_g(nzg, 0.0), w2_z_g(nzg, 0.0);
    MPI_Allreduce(t_sum_z.data(),  t_z_g.data(),  nzg, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
    MPI_Allreduce(w2_sum_z.data(), w2_z_g.data(), nzg, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
    const double inv_nxy = 1.0 / (static_cast<double>(nxg) * nyg);
    const double inv_n   = 1.0 / static_cast<double>(tc_n);
    // wt_sum is dz-weighted (dx3_z), Lz=1 → vol-avg = wt_sum / (nx*ny)
    const double wt_mean = wt_sum * inv_nxy * inv_n;
    const double Nu = 1.0 + wt_mean / alpha_T;
    if (mpi.is_root()) {
      std::printf("[rbc] Nu=%.4f  (wt_mean=%.6f  alpha_T=%.4e  tc_n=%ld)\n", Nu, wt_mean, alpha_T, tc_n);
      std::printf("[rbc] k  T_mean  W_rms\n");
      for (int k = 0; k < nzg; ++k) {
        const double Tk = t_z_g[k]  * inv_nxy * inv_n;
        const double Wk = std::sqrt(std::max(0.0, w2_z_g[k] * inv_nxy * inv_n));
        std::printf("  %3d  %+.5f  %.5f\n", k, Tk, Wk);
      }
    }
  }
  return 0;
}
