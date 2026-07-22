// RBC — GPU backend (G3b hybrid), NOB (variable properties: glycerol).
// Scalar + momentum + properties solved on GPU; pressure solved on CPU (hybrid G3b).
// Gravity along z. Hot bottom +0.5, cold top -0.5. No-slip U,V,W.

#include "rbc_common.hpp"

int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);

  auto is_per = [](const std::string& s){ return s == "periodic"; };
  std::array<int,3>  dims     = {cfg.get<int>("mpi","np1"), cfg.get<int>("mpi","np2"), cfg.get<int>("mpi","np3")};
  std::array<bool,3> periodic = {is_per(cfg.get_or<std::string>("topology","x","periodic")),
                                 is_per(cfg.get_or<std::string>("topology","y","periodic")),
                                 is_per(cfg.get_or<std::string>("topology","z","wall"))};
  core::MpiTopology topo(mpi, dims, periodic);
  std::array<int,3> n_global = {cfg.get<int>("mesh","n1m"), cfg.get<int>("mesh","n2m"), cfg.get<int>("mesh","n3m")};
  core::Subdomain sub(topo, n_global);

  std::array<grid::AxisConfig,3> axes;
  for (int a = 0; a < 3; ++a) {
    const int    is_uni = cfg.get_or<int>   ("uniform_mesh","uniform"+std::to_string(a+1), 1);
    const double gamma  = cfg.get_or<double>("mesh_stretch","gamma" +std::to_string(a+1), 0.0);
    const double L      = cfg.get<double>   ("domain", std::string("L")+char('x'+a));
    axes[a] = {n_global[a], L, is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
  }
  core::Grid grid(sub, axes);
  core::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
  bc.validate();
  auto  tdma_owner = linear_solver::tdma::TdmaRegistry::make_default(topo);
  core::Domain domain{grid, topo, sub, *tdma_owner};

  core::CpuFields fields;
  fields.add(core::Var::U, sub); fields.add(core::Var::V, sub);
  fields.add(core::Var::W, sub); fields.add(core::Var::P, sub);
  fields.add(core::Var::T, sub);

  const double Ra = cfg.get<double>("physics","Ra"), Pr = cfg.get<double>("physics","Pr");
  const double nu = std::sqrt(Pr/Ra), alpha_T = 1.0/std::sqrt(Ra*Pr);
  fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  core::PressureSystem pressure;

  const double DeltaT = cfg.get_or<double>("physics","DeltaT", 0.0);
  physics::CpuPropertyFields props(sub);   // CPU copy for pressure step
  const physics::PropertyModel pmodel = physics::PropertyModel::glycerol(static_cast<real_t>(DeltaT));
  physics::BuoyancyParams buoy;
  buoy.gravity_axis = 2; buoy.coeff = 1; buoy.T_ref = 0;
  buoy.a12pera11 = (std::fabs(pmodel.a[2]) < real_t{1e-14}) ? real_t{0} : pmodel.a[2] / pmodel.a[1];
  buoy.delta_T   = static_cast<real_t>(DeltaT);

  const double t_end    = cfg.get_or<double>("time","t_end",   400.0);
  const real_t dt_cap   = static_cast<real_t>(cfg.get_or<double>("time","dt_start",  0.02));
  const real_t max_cfl  = static_cast<real_t>(cfg.get_or<double>("time","MaxCFL",    0.5));
  const int    n_steps  = cfg.get_or<int>   ("time","n_steps", 200000);
  const int    nmonitor    = cfg.get_or<int>("output","nmonitor",    200);
  const int    nstat_start = cfg.get_or<int>("output","nstat_start", 20000);
  const std::string dir_stats = cfg.get_or<std::string>("output","dir_stats","apps/rbc/statistics/");
  if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
  MPI_Barrier(topo.cart_comm());

  // IC: linear T(z) + multi-mode W perturbation
  {
    const double Lx=cfg.get<double>("domain","Lx"), Ly=cfg.get<double>("domain","Ly"), Lz=cfg.get<double>("domain","Lz");
    const double T_lo=bc.T.face(Direction::Z,Side::Minus).value, T_hi=bc.T.face(Direction::Z,Side::Plus).value;
    const double amp=cfg.get_or<double>("init","perturb",0.02);
    const auto& xc=grid.xc(Direction::X); const auto& yc=grid.xc(Direction::Y); const auto& zc=grid.xc(Direction::Z);
    const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2];
    real_t* W=fields[core::Var::W].data(); real_t* T=fields[core::Var::T].data();
    for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) for (int k=0;k<n3;++k) {
      const int c=(i*n2+j)*n3+k;
      const double z=zc[k], x=xc[i], y=yc[j];
      T[c]=static_cast<real_t>(T_lo+(T_hi-T_lo)*(z/Lz));
      const double env=std::sin(M_PI*z/Lz);
      W[c]+=static_cast<real_t>(amp*env*(std::sin(2*M_PI*x/Lx)*std::cos(2*M_PI*y/Ly)
                                        +std::cos(2*M_PI*x/Lx)*std::sin(2*M_PI*y/Ly)
                                        +std::sin(4*M_PI*x/Lx)*std::sin(4*M_PI*y/Ly)));
    }
  }
  rbc::zero_w_wall(fields[core::Var::W], sub);
  core::sync_field_cpu(fields, core::Var::U, bc, sub); core::sync_field_cpu(fields, core::Var::V, bc, sub);
  core::sync_field_cpu(fields, core::Var::W, bc, sub); core::sync_field_cpu(fields, core::Var::P, bc, sub);
  core::sync_field_cpu(fields, core::Var::T, bc, sub);

  // GPU setup: bind device, allocate GPU counterparts, copy IC to device
  core::bind_gpu_to_local_rank_gpu(mpi);

  core::GpuFields gpu_fields;
  gpu_fields.add(core::Var::U, sub); gpu_fields.add(core::Var::V, sub);
  gpu_fields.add(core::Var::W, sub); gpu_fields.add(core::Var::P, sub);
  gpu_fields.add(core::Var::T, sub);
  gpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  gpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  core::GpuMomentumSystem  gpu_momentum(sub);
  core::GpuScalarSystem    gpu_scalar(sub);
  physics::GpuPropertyFields gpu_props(sub);

  gpu_fields[core::Var::U].to_device(fields[core::Var::U].data());
  gpu_fields[core::Var::V].to_device(fields[core::Var::V].data());
  gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
  gpu_fields[core::Var::P].to_device(fields[core::Var::P].data());
  gpu_fields[core::Var::T].to_device(fields[core::Var::T].data());

  if (mpi.is_root())
    std::printf("[rbc_gpu_nob] Ra=%.2e Pr=%.4g DeltaT=%.1f nu=%.4e alpha=%.4e  n=%dx%dx%d  (GPU hybrid)\n",
                Ra, Pr, DeltaT, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

  // ── time loop ─────────────────────────────────────────────────────────────
  double time=0.0; int step=0;
  rbc::RbcStats stats(sub.n_global()[2]);
  const real_t* dx3_z = grid.dx_ptr(Direction::Z);

  while (time < t_end && step < n_steps) {
    const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);

    physics::update_properties_gpu(gpu_props, gpu_fields[core::Var::T], pmodel);

    equation::solve_scalar_nob_gpu(domain, bc, gpu_fields, gpu_props.KPP, gpu_props.dKPP,
                                   gpu_props.invRhoCp, gpu_props.dinvRhoCp, gpu_scalar, dt);
    core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);

    equation::assemble_momentum_var_visc_gpu(domain, gpu_fields, gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
    physics::add_buoyancy_force_nob_gpu(gpu_momentum, gpu_fields[core::Var::T], gpu_props.invRho, buoy, dt);
    equation::solve_momentum_var_gpu(domain, bc, gpu_fields, gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
    equation::update_velocity_gpu(gpu_fields, gpu_momentum);
    core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);

    // zero_w_wall: MAC face not handled on GPU yet; apply on host then push back
    gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
    rbc::zero_w_wall(fields[core::Var::W], sub);
    gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);

    // Transfer GPU→CPU for CPU pressure solve + diagnostics
    gpu_fields[core::Var::U].to_host(fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_host(fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
    gpu_fields[core::Var::T].to_host(fields[core::Var::T].data());

    // CPU properties needed for NOB pressure projection (invRhoc)
    physics::update_properties_cpu(props, fields[core::Var::T], pmodel);

    equation::solve_pressure_cpu(domain, bc, fields, pressure, dt, props.invRho.data());

    if (DeltaT > 0.0) {
      real_t* t_ptr = fields[core::Var::T].data();
      const std::size_t nf = (std::size_t)sub.n_total()[0]*sub.n_total()[1]*sub.n_total()[2];
      for (std::size_t q = 0; q < nf; ++q)
        t_ptr[q] = std::max(real_t{-0.5}, std::min(real_t{0.5}, t_ptr[q]));
    }

    // Transfer projected velocity + clipped T back to GPU
    gpu_fields[core::Var::U].to_device(fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_device(fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
    gpu_fields[core::Var::T].to_device(fields[core::Var::T].data());

    time += double(dt); ++step;
    rbc::accumulate(stats, domain, sub, fields, dx3_z, nstat_start, step);
    if (step % nmonitor == 0) rbc::print_step(stats, domain, fields, step, time, dt, alpha_T, mpi.is_root());
  }

  rbc::print_final(stats, sub, Ra, Pr, alpha_T, mpi.is_root(), "rbc_gpu_nob");
  return 0;
}
