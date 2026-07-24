// apps/dhvc — GPU backend, OB differentially-heated vertical convection (PaScaL_TCS Fig 7).
//
// Axis map (identical to CPU main.cpp):
//   x = streamwise / vertical / gravity (periodic)  ← buoyancy on U  (gravity_axis=0)
//   y = spanwise                        (periodic)
//   z = wall-normal (hot z=0, cold z=Lz)            (wall)
//
// GPU time loop: scalar → momentum+buoyancy → mass-flow correction → pressure.
// Statistics accumulated via GPU kernels (no full-field D2H during time loop).
// Monitor CSV + stats .dat written from device-reduced data.

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/boundary_ops.hpp"
#include "core/fields.hpp"
#include "equation/momentum/linear_system.hpp"
#include "equation/scalar/linear_system.hpp"
#include "equation/pressure/linear_system.hpp"
#include "core/config.hpp"
#include "core/field_gpu.hpp"
#include "core/gpu/gpu_grid.hpp"
#include "core/gpu_binding.hpp"

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
#include "physics/forcing/forcing.hpp"
#include "driver/cfl.hpp"
#include "physics/properties/properties.hpp"
#include "post/statistics.hpp"
#include "post/statistics_dhvc.hpp"
#include "post/io.hpp"            // write_restart_cpu (checkpoint)

#include <cuda_runtime.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mpmstd;

namespace {

// ── Wall-normal W zeroing kernel ──────────────────────────────────────────────
__global__ void k_zero_w_wall(real_t* W, int n1, int n2, int n3, int h,
                               bool has_lo, bool has_hi) {
    const int ni = n1 - 2*h, nj = n2 - 2*h;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= ni * nj) return;
    const int i = idx / nj + h, j = idx % nj + h;
    if (has_lo) W[(i * n2 + j) * n3 + h]        = real_t(0);
    if (has_hi) W[(i * n2 + j) * n3 + (n3 - h)] = real_t(0);
}


} // anonymous namespace

int main(int argc, char** argv) {
    parallel::mpi::MpiContext mpi(&argc, &argv);
    config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
    const std::string input_path = (argc >= 2) ? argv[1] : "apps/dhvc/input.toml";
    auto cfg = config::Config::load(input_path);

    auto is_per = [](const std::string& s){ return s == "periodic"; };
    std::array<int,3>  dims     = {cfg.get<int>("mpi","np1"), cfg.get<int>("mpi","np2"), cfg.get<int>("mpi","np3")};
    std::array<bool,3> periodic = {is_per(cfg.get_or<std::string>("topology","x","periodic")),
                                   is_per(cfg.get_or<std::string>("topology","y","periodic")),
                                   is_per(cfg.get_or<std::string>("topology","z","wall"))};
    parallel::mpi::MpiTopology topo(mpi, dims, periodic);
    std::array<int,3> n_global = {cfg.get<int>("mesh","n1m"), cfg.get<int>("mesh","n2m"), cfg.get<int>("mesh","n3m")};
    parallel::mpi::Subdomain sub(topo, n_global);

    std::array<grid::AxisConfig,3> axes;
    for (int a = 0; a < 3; ++a) {
        const int    is_uni = cfg.get_or<int>   ("uniform_mesh","uniform"+std::to_string(a+1), 1);
        const double gamma  = cfg.get_or<double>("mesh_stretch","gamma" +std::to_string(a+1), 0.0);
        const double L      = cfg.get<double>   ("domain", std::string("L")+char('x'+a));
        axes[a] = {n_global[a], L, is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
    }
    grid::Grid grid(sub, axes);
    boundary::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
    bc.validate();
    auto  tdma_owner = linear_solver::tdma::TdmaRegistry::make_default(topo);
    core::Domain domain{grid, topo, sub, *tdma_owner};

    const double Ra      = cfg.get<double>("physics","Ra");
    const double Pr      = cfg.get<double>("physics","Pr");
    const double nu      = std::sqrt(Pr / Ra);
    const double alpha_T = 1.0 / std::sqrt(Ra * Pr);

    const double t_end      = cfg.get_or<double>("time","t_end",       400.0);
    const real_t dt_cap     = static_cast<real_t>(cfg.get_or<double>("time","dt_start",  0.02));
    const real_t max_cfl    = static_cast<real_t>(cfg.get_or<double>("time","MaxCFL",    0.5));
    const int    n_steps    = cfg.get_or<int>   ("time","n_steps",    200000);
    const int    nmonitor   = cfg.get_or<int>("output","nmonitor",    500);
    // Statistics keyed on PHYSICAL TIME t (not step) → independent of the variable dt.
    // Accumulate from t>=t_stat_start; write a cumulative-average stats file every
    // t_stat_out of physical time. Accumulation is dt-weighted (true time-average,
    // sampled EVERY step) so it is fully independent of the variable dt.
    const double t_stat_start = cfg.get_or<double>("output","t_stat_start", 30.0);
    const double t_stat_out   = cfg.get_or<double>("output","t_stat_out", 30.0);
    const std::string dir_stats = cfg.get_or<std::string>("output","dir_stats","apps/dhvc/statistics/");
    const std::string restart_out = cfg.get_or<std::string>("restart","dir_out","");  // checkpoint at run end
    const std::string restart_in  = cfg.get_or<std::string>("restart","dir_in","");   // warm-start source (any mesh)
    if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
    MPI_Barrier(topo.cart_comm());

    // CPU cpu_fields: used only for IC setup + stats D2H populate
    core::CpuFields cpu_fields;
    cpu_fields.add(core::Var::U, sub); cpu_fields.add(core::Var::V, sub);
    cpu_fields.add(core::Var::W, sub); cpu_fields.add(core::Var::P, sub);
    cpu_fields.add(core::Var::T, sub);
    cpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
    cpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

    // Initial condition: analytical base state + perturbation (cold start), OR a
    // warm start interpolated from a previous (possibly coarser) run's restart.
    if (restart_in.empty()) {
        const double Lx=cfg.get<double>("domain","Lx"), Ly=cfg.get<double>("domain","Ly"), Lz=cfg.get<double>("domain","Lz");
        const double T_lo=bc.T.face(Direction::Z,Side::Minus).value;
        const double T_hi=bc.T.face(Direction::Z,Side::Plus ).value;
        const double amp=cfg.get_or<double>("init","perturb",0.05);
        const auto& xc=grid.xc(Direction::X); const auto& yc=grid.xc(Direction::Y); const auto& zc=grid.xc(Direction::Z);
        const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2];
        real_t* U=cpu_fields[core::Var::U].data(), *V=cpu_fields[core::Var::V].data();
        real_t* W=cpu_fields[core::Var::W].data(), *T=cpu_fields[core::Var::T].data();
        for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) for (int k=0;k<n3;++k) {
            const int c=(i*n2+j)*n3+k;
            const double z=zc[k], x=xc[i], y=yc[j];
            U[c]=static_cast<real_t>(z*(2.0*z/Lz-1.0)*(z/Lz-1.0)/(12.0*nu));
            T[c]=static_cast<real_t>(T_lo+(T_hi-T_lo)*(z/Lz));
            const double env=std::sin(M_PI*z/Lz);
            const double pert=amp*env*(std::sin(2.0*M_PI*x/Lx)*std::cos(2.0*M_PI*y/Ly)
                                      +std::sin(4.0*M_PI*x/Lx)*std::sin(2.0*M_PI*y/Ly));
            V[c]+=static_cast<real_t>(pert);
            W[c]+=static_cast<real_t>(0.5*amp*env*std::sin(2.0*M_PI*x/Lx));
        }
    } else {
        // WARM START — read the source restart (any mesh) + interpolate (staggered,
        // trilinear, x/y periodic, z clamp) onto this run's grid. U,V,W,T,P(=P_acc).
        const double Lx=cfg.get<double>("domain","Lx"), Ly=cfg.get<double>("domain","Ly"), Lz=cfg.get<double>("domain","Lz");
        const double gz=cfg.get_or<double>("mesh_stretch","gamma3",0.0);
        int sdim[3]={0,0,0};
        if (mpi.is_root()) {
            std::ifstream mf(restart_in + "/meta.txt"); std::string k; double v;
            while (mf>>k>>v) { if(k=="n1m")sdim[0]=(int)v; else if(k=="n2m")sdim[1]=(int)v; else if(k=="n3m")sdim[2]=(int)v; }
        }
        MPI_Bcast(sdim,3,MPI_INT,0,topo.cart_comm());
        if (sdim[0]<=0||sdim[1]<=0||sdim[2]<=0) throw std::runtime_error("warm start: bad/missing meta.txt dims in " + restart_in);
        post::SrcMesh sm = post::make_src_mesh(sdim[0],sdim[1],sdim[2], Lx,Ly,Lz, gz);
        std::vector<real_t> buf;
        const core::Var vars[5]={core::Var::U,core::Var::V,core::Var::W,core::Var::T,core::Var::P};
        const char*     names[5]={"U","V","W","T","P"};
        const int       stag[5] ={0,1,2,-1,-1};   // U:x-face V:y-face W:z-face T,P:center

        // I/O path: "slab" (Tier 1, default) reads only this rank's source z-slab via
        // collective MPI-IO (per-rank memory ~ full_field/N_z, no Bcast); "replicate"
        // (legacy) reads the whole field on every rank + Bcast. Both feed the IDENTICAL
        // trilinear interpolation — only the source z indexing differs.
        const std::string ws_io = cfg.get_or<std::string>("restart","warmstart_io","slab");
        const bool use_slab = (ws_io != "replicate");
        int kz0 = 0, slab_n3 = -1;
        if (use_slab) {
            // This rank's target z-extent (interior faces bound every staggered node).
            const int n3t = static_cast<int>(sub.n_total()[2]), h = kHaloWidth;
            const auto& tzf = grid.xf(Direction::Z);
            const double z_min = static_cast<double>(tzf[h]);
            const double z_max = static_cast<double>(tzf[n3t - h]);
            post::compute_src_z_box(sm, z_min, z_max, /*margin=*/2, kz0, slab_n3);
        }
        for (int q=0;q<5;++q) {
            const std::string fpath = restart_in + "/" + std::string(names[q]) + ".bin";
            if (use_slab) post::read_source_slab(buf, sdim[0],sdim[1],sdim[2], kz0, slab_n3, fpath, topo.cart_comm());
            else          post::read_global_field(buf, sdim[0],sdim[1],sdim[2], fpath, topo.cart_comm());
            post::interp_global_to_field(cpu_fields[vars[q]], grid, buf, sm, stag[q], kz0, slab_n3);
        }
        if (mpi.is_root())
            std::printf("[warm-start] interpolated %dx%dx%d -> %dx%dx%d from %s (io=%s)\n",
                        sdim[0],sdim[1],sdim[2], n_global[0],n_global[1],n_global[2],
                        restart_in.c_str(), use_slab ? "slab" : "replicate");

        // dry-run: dump per-var interior checksums (for slab-vs-replicate A/B), then exit.
        if (cfg.get_or<int>("restart","warmstart_dryrun",0)) {
            const auto nt=sub.n_total(); const int n1=nt[0],n2=nt[1],n3=nt[2],h=kHaloWidth;
            if (mpi.is_root()) std::printf("[warm-start dryrun] var: sum  sumsq  min  max  (global interior, io=%s)\n", use_slab?"slab":"replicate");
            for (int q=0;q<5;++q) {
                const real_t* d = cpu_fields[vars[q]].data();
                double s=0, s2=0, mn=1e300, mx=-1e300;
                for (int i=h;i<n1-h;++i) for (int j=h;j<n2-h;++j) for (int k=h;k<n3-h;++k) {
                    const double v=(double)d[((std::size_t)i*n2+j)*n3+k];
                    s+=v; s2+=v*v; mn=std::min(mn,v); mx=std::max(mx,v);
                }
                double gs,gs2,gmn,gmx;
                MPI_Reduce(&s,&gs,1,MPI_DOUBLE,MPI_SUM,0,topo.cart_comm());
                MPI_Reduce(&s2,&gs2,1,MPI_DOUBLE,MPI_SUM,0,topo.cart_comm());
                MPI_Reduce(&mn,&gmn,1,MPI_DOUBLE,MPI_MIN,0,topo.cart_comm());
                MPI_Reduce(&mx,&gmx,1,MPI_DOUBLE,MPI_MAX,0,topo.cart_comm());
                if (mpi.is_root()) std::printf("[warm-start dryrun] %s: %.15e %.15e %.15e %.15e\n", names[q],gs,gs2,gmn,gmx);
            }
            MPI_Barrier(topo.cart_comm());
            return 0;   // RAII (topo, then mpi=MpiContext) frees comm + MPI_Finalize on return
        }
    }
    {
        const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2], h=kHaloWidth;
        real_t* W=cpu_fields[core::Var::W].data();
        const auto& ztopo=sub.topology().axis(Direction::Z);
        if (ztopo.west_rank==MPI_PROC_NULL)
            for(int i=0;i<n1;++i) for(int j=0;j<n2;++j) W[(i*n2+j)*n3+h]=0;
        if (ztopo.east_rank==MPI_PROC_NULL)
            for(int i=0;i<n1;++i) for(int j=0;j<n2;++j) W[(i*n2+j)*n3+(n3-h)]=0;
    }
    core::sync_field_cpu(cpu_fields, core::Var::U, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::V, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::W, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::P, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::T, bc, sub);

    // GPU setup
    core::bind_gpu_to_local_rank_gpu(mpi);
    core::GpuGridArrays gpu_grid_arrays(grid);
    domain.gpu_grid = &gpu_grid_arrays;

    core::GpuFields gpu_fields;
    gpu_fields.add(core::Var::U, sub); gpu_fields.add(core::Var::V, sub);
    gpu_fields.add(core::Var::W, sub); gpu_fields.add(core::Var::P, sub);
    gpu_fields.add(core::Var::T, sub);
    gpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
    gpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

    gpu_fields[core::Var::U].to_device(cpu_fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_device(cpu_fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_device(cpu_fields[core::Var::W].data());
    gpu_fields[core::Var::P].to_device(cpu_fields[core::Var::P].data());
    gpu_fields[core::Var::T].to_device(cpu_fields[core::Var::T].data());

    equation::momentum::GpuLinearSystem gpu_momentum(sub);
    equation::scalar::GpuLinearSystem   gpu_scalar(sub);
    equation::pressure::LinearSystem    pressure;

    // Accumulated pressure Pⁿ (van Kan) lives in Var::P — already seeded by the
    // gpu_fields[Var::P] upload above (0 on cold start, interpolated P on warm-
    // start). The per-step increment δp is owned by the pressure LinearSystem.

    // OB unit property cpu_fields for NOB var kernels (mu=1, inv_rho=1, kappa=1,
    // dkappa=0, inv_rhocp=1, dinv_rhocp=0).  With DeltaT=0 the default
    // PropertyModel evaluates to unit constants on every cell → the var kernels
    // reduce to the correct PaScaL_TCS OB scheme with semi-implicit convection
    // and −M·q^n residual correction, fixing the dt collapse at Ra=1e7.
    physics::properties::GpuPropertyFields gpu_properties(sub);
    physics::properties::update_properties_gpu(gpu_properties, gpu_fields[core::Var::T],
                                   physics::properties::PropertyModel{});

    // Warm start: the interpolated velocity is not discretely divergence-free →
    // one pressure projection cleans it before time marching (methodology step 4).
    if (!restart_in.empty()) {
        const real_t dt0 = static_cast<real_t>(cfg.get_or<double>("time","dt_start",0.001));
        equation::pressure::solve_pressure_constant_property_gpu(domain, bc, gpu_fields, pressure, dt0);
        if (mpi.is_root())
            std::printf("[warm-start] projected interpolated field divergence-free (dt0=%.2e)\n", dt0);
    }

    // GPU statistics (device reductions + monitor.csv + stats_t*.dat, no full-field D2H).
    post::DhvcStatisticsConfig stat_cfg;
    stat_cfg.nmonitor     = nmonitor;     stat_cfg.dir_stats  = dir_stats;
    stat_cfg.t_stat_start = t_stat_start; stat_cfg.t_stat_out = t_stat_out;
    stat_cfg.Ra = Ra; stat_cfg.nu = nu;   stat_cfg.alpha_T = alpha_T;
    post::DhvcStatistics statistics(sub, grid, gpu_grid_arrays, domain, stat_cfg,
                                    topo.cart_comm(), mpi.is_root());

    physics::buoyancy::BuoyancyParams buoyancy_params;
    buoyancy_params.gravity_axis = 0; buoyancy_params.coeff=1.0; buoyancy_params.T_ref=0.0;
    buoyancy_params.a12pera11=0.0; buoyancy_params.delta_T=0.0;

    const double total_vol = physics::forcing::channel_total_volume_cpu(domain);
    const auto& ztopo = sub.topology().axis(Direction::Z);
    const bool has_lo = (ztopo.west_rank == MPI_PROC_NULL);
    const bool has_hi = (ztopo.east_rank == MPI_PROC_NULL);

    if (mpi.is_root())
        std::printf("[dhvc_gpu] Ra=%.2e Pr=%.3g nu=%.4e alpha=%.4e n=%dx%dx%d GPU\n",
                    Ra, Pr, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

    // ── time loop ─────────────────────────────────────────────────────────────
    double time = 0.0; int step = 0;
    double net_dpdx = 0.0;
    double t_ss = 0.0; int n_time_steps = 10;
    auto t00 = std::chrono::high_resolution_clock::now();
    using hrc = std::chrono::high_resolution_clock;
    auto ms_since = [](hrc::time_point t) {
        return std::chrono::duration<double,std::milli>(hrc::now()-t).count();
    };
    double ts_cfl=0, ts_scalar=0, ts_mom=0, ts_corr=0, ts_sync=0, ts_pres=0;
    // Whole-run compute timer (never reset, unlike t00) + per-monitor-interval rate.
    MPI_Barrier(topo.cart_comm());
    const auto t_loop_start = hrc::now();
    auto t_last_mon = t_loop_start;
    int  last_mon_step = 0;

    while (time < t_end && step < n_steps) {
        const bool do_timing = (step < n_time_steps);
        hrc::time_point tPrev;

        if (do_timing) { cudaDeviceSynchronize(); tPrev = hrc::now(); }
        const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);
        if (do_timing) { cudaDeviceSynchronize(); ts_cfl += ms_since(tPrev); tPrev = hrc::now(); }

        // (1) Scalar: T^n → T^{n+1}  (NOB var path with unit cpu_properties = PaScaL_TCS scheme)
        equation::scalar::solve_scalar_variable_property_gpu(domain, bc, gpu_fields,
            gpu_properties.KPP, gpu_properties.dKPP, gpu_properties.invRhoCp, gpu_properties.dinvRhoCp,
            gpu_scalar, dt);
        core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);
        if (do_timing) { cudaDeviceSynchronize(); ts_scalar += ms_since(tPrev); tPrev = hrc::now(); }

        // (2) Momentum predictor + OB buoyancy  (NOB var path with unit cpu_properties)
        equation::momentum::assemble_momentum_variable_property_gpu(domain, gpu_fields,
            gpu_properties.Mu, gpu_properties.invRho, gpu_momentum, dt);
        physics::buoyancy::add_buoyancy_force_constant_property_gpu(domain, gpu_momentum, gpu_fields[core::Var::T], buoyancy_params, dt);
        equation::pressure::add_pressure_gradient_constant_property_gpu(domain, gpu_momentum, gpu_fields[core::Var::P], dt);  // incremental: −dt·∇Pⁿ
        equation::momentum::solve_momentum_variable_property_gpu(domain, bc, gpu_fields,
            gpu_properties.Mu, gpu_properties.invRho, gpu_momentum, dt);
        equation::momentum::update_velocity_gpu(gpu_fields, gpu_momentum);
        if (do_timing) { cudaDeviceSynchronize(); ts_mom += ms_since(tPrev); tPrev = hrc::now(); }

        // (3) Zero net streamwise flux
        physics::forcing::apply_mass_flow_correction_gpu(domain, gpu_fields, 0.0, double(dt), total_vol, net_dpdx);
        if (do_timing) { cudaDeviceSynchronize(); ts_corr += ms_since(tPrev); tPrev = hrc::now(); }

        core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);
        core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
        core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);

        // (4) Zero W at wall faces (GPU kernel, no D2H)
        {
            const auto nt=sub.n_total();
            const int n1=nt[0], n2=nt[1], n3=nt[2], h=kHaloWidth;
            const int block=256, grid_xy=((n1-2*h)*(n2-2*h)+block-1)/block;
            k_zero_w_wall<<<grid_xy,block>>>(gpu_fields[core::Var::W].data(),
                                             n1, n2, n3, h, has_lo, has_hi);
            core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
        }
        if (do_timing) { cudaDeviceSynchronize(); ts_sync += ms_since(tPrev); tPrev = hrc::now(); }

        // (5) Pressure
        equation::pressure::solve_pressure_constant_property_gpu(domain, bc, gpu_fields, pressure, dt);
        equation::pressure::accumulate_pressure_gpu(gpu_fields, pressure);  // incremental: Pⁿ⁺¹ = Pⁿ + δp
        if (do_timing) { cudaDeviceSynchronize(); ts_pres += ms_since(tPrev); }

        time += double(dt); ++step;

        // (6) GPU statistics accumulation — NO D2H
        statistics.accumulate_if_due(time, dt, gpu_fields);   // dt-weighted, every step ≥ t_stat_start

        // (7) Monitor: ~1.6 KB D2H, write monitor.csv (+ live wall-clock / ms-per-step)
        if (step % nmonitor == 0) {
            cudaDeviceSynchronize();                          // all step work done → wall is exact
            const double cum_wall_s     = ms_since(t_loop_start) / 1000.0;
            const double recent_ms_step = ms_since(t_last_mon) / std::max(1, step - last_mon_step);
            statistics.monitor_if_due(step, time, dt, gpu_fields, cum_wall_s, recent_ms_step);
            t_last_mon = hrc::now(); last_mon_step = step;
        }

        // (8) Stats: cumulative-average profile file-out every t_stat_out of PHYSICAL
        //     TIME from t_stat_start, so QoI convergence is trackable from the .dat seq.
        statistics.write_profile_if_due(time, step);

        // Timing: first 10 steps → per-step average + estimated total runtime
        if (step == n_time_steps && mpi.is_root()) {
            const double ms_per_step = ms_since(t00) / n_time_steps;
            const double est_total_h = ms_per_step * (t_end / std::max(double(dt), 1e-9)) / 3600000.0;
            std::printf("[TIMING] %d steps  avg=%.1f ms/step\n", n_time_steps, ms_per_step);
            std::printf("[TIMING_BREAKDOWN] per step:  cfl=%.1f  scalar=%.1f  mom=%.1f  corr=%.1f  sync=%.1f  pres=%.1f  ms\n",
                        ts_cfl/n_time_steps, ts_scalar/n_time_steps, ts_mom/n_time_steps,
                        ts_corr/n_time_steps, ts_sync/n_time_steps, ts_pres/n_time_steps);
            std::printf("[TIMING] t_end=%.0f  rough total: %.2f h (n_steps cap=%d)\n",
                        t_end, est_total_h, n_steps);
            std::printf("[TIMING] stats out: from t=%.1f every t=%.1f (to t_end=%.0f)\n",
                        t_stat_start, t_stat_out, t_end);
            t00 = hrc::now(); t_ss = 0.0;
        }
    }

    if (step > n_time_steps && mpi.is_root())
        std::printf("[TIMING_SS] %d steps (steady-state)  total=%.1f ms/step\n",
                    step - n_time_steps, ms_since(t00)/(step - n_time_steps));

    // ── total compute time for the whole run (loop start → here) ─────────────────
    cudaDeviceSynchronize();
    MPI_Barrier(topo.cart_comm());
    const double total_s = ms_since(t_loop_start) / 1000.0;
    if (mpi.is_root()) {
        const double avg_ms = total_s * 1000.0 / std::max(1, step);
        std::printf("[TIMING_TOTAL] compute = %.1f s (%.2f h)  |  %d steps  |  avg %.2f ms/step\n",
                    total_s, total_s / 3600.0, step, avg_ms);
    }


    // ── checkpoint write (restart file at run end, t=t_end) ──────────────────────
    // U,V,W,T + P (= accumulated pressure P_acc for the incremental scheme). Global
    // C-order MPI-IO (decomposition-independent). meta.txt carries t/step/mesh/Ra for
    // continuation + the (future) cross-mesh interpolation warm-start.
    if (!restart_out.empty()) {
        core::copy_all_to_host(gpu_fields, cpu_fields);   // U/V/W/T/P device→host (P.bin = accumulated pressure)
        std::error_code ec; std::filesystem::create_directories(restart_out, ec);
        MPI_Barrier(topo.cart_comm());
        post::write_restart_gathered_cpu(cpu_fields[core::Var::U], sub, restart_out + "/U.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::V], sub, restart_out + "/V.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::W], sub, restart_out + "/W.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::T], sub, restart_out + "/T.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::P], sub, restart_out + "/P.bin");
        if (mpi.is_root()) {
            std::ofstream mf(restart_out + "/meta.txt");
            mf << std::setprecision(17)
               << "step " << step << "\n" << "time " << time << "\n"
               << "n1m " << n_global[0] << "\n" << "n2m " << n_global[1] << "\n"
               << "n3m " << n_global[2] << "\n" << "Ra " << Ra << "\n";
            std::printf("[restart] wrote %s  (step=%d t=%.4f; U/V/W/T/P + meta)\n",
                        restart_out.c_str(), step, time);
        }
    }

    const double re_dstar = statistics.final_re_delta_star();
    if (mpi.is_root())
        std::printf("[dhvc_gpu] done: Ra=%.2e  Re_dstar=%.4f  (target=%.4f)\n",
                    Ra, re_dstar, 0.23*std::pow(Ra, 0.28));
    return 0;
}
