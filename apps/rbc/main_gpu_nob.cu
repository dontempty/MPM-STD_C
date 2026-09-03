// RBC — GPU backend (G3b hybrid), NOB (variable properties: glycerol).
// Scalar + momentum + properties solved on GPU; pressure solved on CPU (hybrid G3b).
// Gravity along z. Hot bottom +0.5, cold top -0.5. No-slip U,V,W.

#include "rbc_common.hpp"
using namespace mpmstd;
#include "core/gpu/gpu_grid.hpp"
#include "post/statistics_rbc.hpp"
#include "post/timing.hpp"
#include <memory>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

namespace {
using hrc = std::chrono::high_resolution_clock;
inline double ms_since(hrc::time_point t0) {
  return std::chrono::duration<double,std::milli>(hrc::now()-t0).count();
}

// GPU kernel: zero W at wall faces (replaces D2H+CPU zero_w_wall+H2D each step)
__global__ void k_zero_w_wall_gpu(double* W, int n2t, int n3t, int h,
                                   int n1t, bool has_lower, bool has_upper) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n1t || j >= n2t) return;
  if (has_lower) W[(i*n2t+j)*n3t + h]        = 0.0;
  if (has_upper) W[(i*n2t+j)*n3t + (n3t - h)] = 0.0;
}

// GPU kernel: clamp T to [-0.5, 0.5] on INTERIOR cells only. The z-wall ghost
// planes hold the antisymmetric Dirichlet mirror 2*T_wall - T_in, which
// legitimately exceeds the physical range (0.5155 at the hot wall). Clipping
// ghosts pinned the effective wall at the ghost CENTER (half a cell outside
// the face) and halved the wall conductance every step — measured as the 2x
// wall-gap flux kink in the mean T profile (Report/31).
// clip_count: running counter of clipped cells (audit D3, Report/RBC — the
// clip is not in the paper and acts as a non-conservative heat sink/source;
// if it fires at statistical steady state it can bias theta_c, so count it).
__global__ void k_clip_T_gpu(double* T, int n1i, int n2i, int n3i,
                             int n2t, int n3t, unsigned long long* clip_count) {
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.z * blockDim.z + threadIdx.z;
  if (ii >= n1i || jj >= n2i || kk >= n3i) return;
  const std::size_t c = (((std::size_t)(ii + kHaloWidth)) * n2t + (jj + kHaloWidth)) * n3t
                      + (kk + kHaloWidth);
  const double v = T[c];
  if (v < -0.5 || v > 0.5) {
    T[c] = v < -0.5 ? -0.5 : 0.5;
    atomicAdd(clip_count, 1ULL);
  }
}

} // anonymous namespace


//----------------------------------------------- Main -----------------------------------------------
int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);

  // ═══════════════════════════════════════════════════════════════════════════
  //  [입력] 모든 config 값을 여기서 한 번에 읽는다. 아래 조립 단계는 이 값들만 쓴다.
  //  cfg 는 불변 문자열 맵이라 읽는 순서는 결과에 영향 없음(필수 키는 없으면 throw).
  // ═══════════════════════════════════════════════════════════════════════════
  auto is_per = [](const std::string& s){ return s == "periodic"; };
  // [mpi] / [topology] / [mesh]
  const std::array<int,3>  dims     = {cfg.get<int>("mpi","np1"), cfg.get<int>("mpi","np2"), cfg.get<int>("mpi","np3")};
  const std::array<bool,3> periodic = {is_per(cfg.get_or<std::string>("topology","x","periodic")),
                                       is_per(cfg.get_or<std::string>("topology","y","periodic")),
                                       is_per(cfg.get_or<std::string>("topology","z","wall"))};
  const std::array<int,3>  n_global = {cfg.get<int>("mesh","n1m"), cfg.get<int>("mesh","n2m"), cfg.get<int>("mesh","n3m")};
  // [domain] / [uniform_mesh] / [mesh_stretch] — 축별 메쉬(도메인 길이·균일여부·tanh γ)
  const double Lx = cfg.get<double>("domain","Lx"), Ly = cfg.get<double>("domain","Ly"), Lz = cfg.get<double>("domain","Lz");
  const double L_axis[3] = {Lx, Ly, Lz};
  std::array<grid::AxisConfig,3> axes;
  for (int a = 0; a < 3; ++a) {
    const int    is_uni = cfg.get_or<int>   ("uniform_mesh","uniform"+std::to_string(a+1), 1);
    const double gamma  = cfg.get_or<double>("mesh_stretch","gamma" +std::to_string(a+1), 0.0);
    axes[a] = {n_global[a], L_axis[a], is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
  }
  // [physics] — Ra/Pr에서 free-fall 무차원 nu, alpha_T 유도
  const double Ra = cfg.get<double>("physics","Ra"), Pr = cfg.get<double>("physics","Pr");
  const double nu = std::sqrt(Pr/Ra), alpha_T = 1.0/std::sqrt(Ra*Pr);
  const double DeltaT       = cfg.get_or<double>("physics","DeltaT", 0.0);
  const int    gravity_axis = cfg.get_or<int>("physics","gravity_axis", 2);
  // gravity_sign: +1 (default) = validated bottom-hot RBC; -1 flips gravity for
  // the exact z-mirror (top-hot) symmetry test. No-op at the default.
  const double gravity_sign = cfg.get_or<double>("physics","gravity_sign", 1.0);
  // [time]
  const double t_end   = cfg.get_or<double>("time","t_end",   400.0);
  const real_t dt_cap  = static_cast<real_t>(cfg.get_or<double>("time","dt_start",  0.02));
  const real_t max_cfl = static_cast<real_t>(cfg.get_or<double>("time","MaxCFL",    0.5));
  const int    n_steps = cfg.get_or<int>   ("time","n_steps", 200000);
  // [output] — 시간 기반 통계: 물리시간 t_stat_start부터 누적, t_stat_out 간격마다 누적평균 기록
  //            (스텝 기반 창은 dt 의존이라 케이스 간 불공정했음)
  const int    nmonitor     = cfg.get_or<int>("output","nmonitor",    200);
  const double t_stat_start = cfg.get_or<double>("output","t_stat_start", 7500.0);
  const double t_stat_out   = cfg.get_or<double>("output","t_stat_out",   2500.0);
  const std::string dir_stats = cfg.get_or<std::string>("output","dir_stats","apps/rbc/statistics/");
  // [init]
  const double amp = cfg.get_or<double>("init","perturb",0.02);
  // [restart]
  const std::string restart_in  = cfg.get_or<std::string>("restart","dir_in",  "");
  const std::string restart_out = cfg.get_or<std::string>("restart","dir_out", "");
  // nrestart_write: 주기적 체크포인트 간격(스텝; 0=끔, 최종만). ckpt_a/ckpt_b 교대 기록으로
  //   쓰기 중 잡이 죽어도 한 벌은 온전; 재시작은 stats.txt step이 더 큰 ckpt를 dir_in으로.
  const int  nrestart_write = cfg.get_or<int>("restart","nrestart_write", 0);
  // stats_reset=1: 이 재시작 지점부터 러닝 평균을 완전히 새로 시작(누적 tc_sum/tc_n/wt_sum 무시)
  //   — 옛 이력이 새 누적 창을 희석/지연시키는 문제(Report/35) 회피.
  const bool stats_reset    = cfg.get_or<int>("restart", "stats_reset", 0) != 0;

  // ═══════════════════════════════════════════════════════════════════════════
  //  [조립] 위 입력값으로 객체를 만든다 (스칼라 cfg 읽기는 위에서 끝; BC만 전용
  //  로더 load_problem_from_config 가 [topology]/[bc.*] 섹션을 직접 해석).
  // ═══════════════════════════════════════════════════════════════════════════
  parallel::mpi::MpiTopology topo(mpi, dims, periodic);
  parallel::mpi::Subdomain sub(topo, n_global);
  grid::Grid grid(sub, axes);
  boundary::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
  bc.validate();
  auto  tdma_owner = linear_solver::tdma::TdmaRegistry::make_default(topo);
  core::Domain domain{grid, topo, sub, *tdma_owner};

  core::CpuFields cpu_fields;
  cpu_fields.add(core::Var::U, sub); cpu_fields.add(core::Var::V, sub);
  cpu_fields.add(core::Var::W, sub); cpu_fields.add(core::Var::P, sub);
  cpu_fields.add(core::Var::T, sub);

  cpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  cpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  physics::properties::CpuPropertyFields cpu_properties(sub);   // CPU copy for pressure step
  const auto property_model  = physics::properties::glycerol(static_cast<real_t>(DeltaT));
  const auto buoyancy_params = physics::buoyancy::BuoyancyParams::from_property_model(property_model, gravity_axis, static_cast<real_t>(gravity_sign));

  if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
  MPI_Barrier(topo.cart_comm());

  // IC: linear T(z) + multi-mode W perturbation, or restart from file
  post::Restart restart(restart_in, restart_out, nrestart_write);
  post::RestartState restart_state;
  if (!restart.warm()) {
    const double T_lo=bc.T.face(Direction::Z,Side::Minus).value, T_hi=bc.T.face(Direction::Z,Side::Plus).value;
    const auto& xc=grid.xc(Direction::X); const auto& yc=grid.xc(Direction::Y); const auto& zc=grid.xc(Direction::Z);
    const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2];
    real_t* W=cpu_fields[core::Var::W].data(); real_t* T=cpu_fields[core::Var::T].data();
    for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) for (int k=0;k<n3;++k) {
      const int c=(i*n2+j)*n3+k;
      const double z=zc[k], x=xc[i], y=yc[j];
      T[c]=static_cast<real_t>(T_lo+(T_hi-T_lo)*(z/Lz));
      const double env=std::sin(M_PI*z/Lz);
      W[c]+=static_cast<real_t>(amp*env*(std::sin(2*M_PI*x/Lx)*std::cos(2*M_PI*y/Ly)
                                        +std::cos(2*M_PI*x/Lx)*std::sin(2*M_PI*y/Ly)
                                        +std::sin(4*M_PI*x/Lx)*std::sin(4*M_PI*y/Ly)));
    }
    rbc::zero_w_wall(cpu_fields[core::Var::W], sub);
    core::sync_field_cpu(cpu_fields, core::Var::U, bc, sub); core::sync_field_cpu(cpu_fields, core::Var::V, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::W, bc, sub); core::sync_field_cpu(cpu_fields, core::Var::P, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::T, bc, sub);
  } else {
    restart_state = restart.load(cpu_fields, bc, sub, topo.cart_comm());
    core::sync_field_cpu(cpu_fields, core::Var::P, bc, sub);
  }

  // GPU setup: bind device, upload grid metrics, allocate GPU cpu_fields
  core::bind_gpu_to_local_rank_gpu(mpi);
  core::GpuGridArrays gpu_grid_arrays(grid);
  domain.gpu_grid = &gpu_grid_arrays;

  core::GpuFields gpu_fields;
  gpu_fields.add(core::Var::U, sub); gpu_fields.add(core::Var::V, sub);
  gpu_fields.add(core::Var::W, sub); gpu_fields.add(core::Var::P, sub);
  gpu_fields.add(core::Var::T, sub);
  gpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  gpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  equation::scalar::GpuLinearSystem    gpu_scalar(sub);
  equation::momentum::GpuLinearSystem  gpu_momentum(sub);
  equation::pressure::LinearSystem pressure;
  physics::properties::GpuPropertyFields gpu_properties(sub);

  gpu_fields[core::Var::U].to_device(cpu_fields[core::Var::U].data());
  gpu_fields[core::Var::V].to_device(cpu_fields[core::Var::V].data());
  gpu_fields[core::Var::W].to_device(cpu_fields[core::Var::W].data());
  gpu_fields[core::Var::P].to_device(cpu_fields[core::Var::P].data());
  gpu_fields[core::Var::T].to_device(cpu_fields[core::Var::T].data());

  // Incremental (van Kan) projection: persistent accumulated pressure Pⁿ.
  // Predictor subtracts dt·(1/ρ̂)∇Pⁿ; the Poisson then solves the increment δp
  // (Var::P) and Pⁿ⁺¹ = Pⁿ + δp. Seeded from the restart P (zero when absent —
  // self-starting: the first solve returns the full pressure). Report/32.

  if (mpi.is_root())
    std::printf("[rbc_gpu_nob] Ra=%.2e Pr=%.4g DeltaT=%.1f nu=%.4e alpha=%.4e  n=%dx%dx%d  (GPU hybrid)\n",
                Ra, Pr, DeltaT, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

  cfg.echo(mpi.is_root());
  cfg.warn_unused_keys(mpi.is_root());   // config honesty (stage 4): flag unread (typo?) keys
  // ── time loop ─────────────────────────────────────────────────────────────
  double time = restart_state.time; int step = restart_state.step;
  post::RbcStatisticsConfig stat_cfg;
  stat_cfg.t_stat_start = t_stat_start; stat_cfg.t_stat_out = t_stat_out;
  stat_cfg.nmonitor     = nmonitor;     stat_cfg.dir_stats  = dir_stats;
  stat_cfg.Ra = Ra; stat_cfg.Pr = Pr;   stat_cfg.DeltaT = DeltaT; stat_cfg.alpha_T = alpha_T;
  post::RbcStatistics statistics(sub, gpu_grid_arrays, grid, stat_cfg, topo.cart_comm(), mpi.is_root());
  statistics.import_extras(restart_state, stats_reset);   // running sums from restart (unless stats_reset)

  // ── scaling-run timing (apps/rbc/timing_plan.md + timing_solve_plan.md) ──
  // Measurement window = loop iterations [warmup, warmup+measure) counted from
  // THIS launch (restart-independent). timing measure=0 (default) → fully off.
  const int  timing_warmup   = cfg.get_or<int>("timing","warmup", 20);
  const int  timing_measure  = cfg.get_or<int>("timing","measure", 0);
  const bool timing_detailed = cfg.get_or<int>("timing","detailed", 0) != 0;
  const std::string timing_dir = cfg.get_or<std::string>("timing","dir", "apps/rbc/timing/");

  // Top-level events marked here in main; the four instrumented library calls
  // append their own blocks (names exported from their public headers).
  enum MainTimingEvent {
    kMainTimingCflDt = 0, kMainTimingProperties, kMainTimingScalarHalo,
    kMainTimingBuoyancy, kMainTimingPressureGradient, kMainTimingVelocityUpdate,
    kMainTimingVelocityBoundary, kMainTimingPressureUpdate,
    kMainTimingTemperatureClip, kMainTimingStatistics, kMainTimingEventCount
  };
  std::vector<std::string> timing_event_names = {
    "cfl_dt", "properties", "scalar_halo", "momentum_buoyancy",
    "momentum_pressure_gradient", "momentum_velocity_update",
    "velocity_boundary", "pressure_update", "temperature_clip", "statistics" };
  const int timing_scalar_base = (int)timing_event_names.size();
  for (auto* name : equation::scalar::kGpuTimingEventNames) timing_event_names.push_back(name);
  const int timing_assemble_base = (int)timing_event_names.size();
  for (auto* name : equation::momentum::kGpuAssembleTimingEventNames) timing_event_names.push_back(name);
  const int timing_momentum_base = (int)timing_event_names.size();
  for (auto* name : equation::momentum::kGpuSolveTimingEventNames) timing_event_names.push_back(name);
  const int timing_pressure_base = (int)timing_event_names.size();
  for (auto* name : equation::pressure::kGpuPressureTimingEventNames) timing_event_names.push_back(name);
  const int timing_marks_per_step = 12
      + equation::scalar::kGpuTimingMarksPerCall            // 11
      + equation::momentum::kGpuAssembleTimingMarksPerCall  // 3
      + equation::momentum::kGpuSolveTimingMarksPerCall     // 41
      + equation::pressure::kGpuPressureTimingMarksPerCall; // 13  → 80 total
  std::unique_ptr<post::TimingRecorder> timing_recorder;
  if (timing_measure > 0)
    timing_recorder = std::make_unique<post::TimingRecorder>(
        timing_measure, timing_marks_per_step, timing_event_names,
        topo.cart_comm(), timing_detailed);
  post::TimingRecorder* rec = nullptr;        // non-null only inside the window
  auto main_mark = [&](int event) { if (rec) rec->mark(event); };
  int  timing_iteration = 0;
  bool timing_window_done = false;

  // ── 한 스텝의 흐름 ──────────────────────────────────────────────────────────
  //  [1] CFL → dt   [2] 물성 갱신(Tⁿ)   [3] 에너지 방정식 → Tⁿ⁺¹
  //  [4] 물성 재갱신(Tⁿ⁺¹)   [5] 운동량 예측자 → U*,V*,W*   [6] 벽면 W=0
  //  [7] 압력 Poisson + 투영 → 발산 제거, Pⁿ⁺¹=Pⁿ+δp   [8] T 클리핑(NOB)
  //  [9] 통계 누적 / 모니터 출력 / 주기적 체크포인트
  while (time < t_end && step < n_steps) {
    if (timing_recorder) {
      if (timing_iteration == timing_warmup) timing_recorder->start_window();
      const int measured = timing_iteration - timing_warmup;
      rec = (measured >= 0 && measured < timing_measure) ? timing_recorder.get() : nullptr;
      if (rec) rec->begin_iteration(measured);
    }

    // [1] 적응 시간간격: 현재 속도장의 CFL 조건으로 dt = min(MaxCFL/속도, dt_cap)
    const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);
    main_mark(kMainTimingCflDt);
    // [2] NOB 변물성 갱신: 현재 온도 Tⁿ으로 μ, 1/ρ, κ, 1/(ρcp) 등을 다항식(글리세롤 모델)으로 재계산
    //     — 아래 에너지 방정식이 이 Tⁿ 물성을 사용
    physics::properties::update_properties_gpu(gpu_properties, gpu_fields[core::Var::T], property_model);
    main_mark(kMainTimingProperties);

    // [3] 에너지(온도) 방정식 — 대류 + 변수-κ 확산, 준음해 델타-폼:
    //     solve 안에서 sweep마다 A 밴드와 RHS를 동시에 조립(k_build_A_rhs_nob, 융합 build_A_RHS;
    //     stage-1이 명시 잔차 R을 인라인 계산) 후 3-sweep ADI(x,y,z 삼중대각 TDMA)로 δT를 풀어 T += δT
    //     sync_field = MPI 할로 교환 + 경계조건 재적용 (Tⁿ⁺¹ 완성)
    cudaDeviceSynchronize();
    equation::scalar::solve_scalar_variable_property_gpu(domain, bc, gpu_fields, gpu_properties.KPP, gpu_properties.dKPP,
                                   gpu_properties.invRhoCp, gpu_properties.dinvRhoCp, gpu_scalar, dt,
                                   rec, timing_scalar_base);
    core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);
    cudaDeviceSynchronize();
    main_mark(kMainTimingScalarHalo);

    // [4] 물성 재갱신 — 방금 푼 Tⁿ⁺¹로 μ, 1/ρ를 다시 계산해 운동량/부력이 Tⁿ⁺¹ 물성을 쓰게 함
    // Re-evaluate properties from the just-solved T^{n+1} so momentum viscosity/density
    // (mu, inv_rho) and the NOB buoyancy inv_rho use T^{n+1} — matches PaScaL_TCS
    // momentum_coeffi(T) called AFTER thermal_solver (MPM-STD staggered-time scheme).
    // Energy above already used the T^n properties from line 408.
    physics::properties::update_properties_gpu(gpu_properties, gpu_fields[core::Var::T], property_model);
    main_mark(kMainTimingProperties);

    // [5] 운동량 예측자(predictor) — 중간 속도 U*,V*,W* 계산:
    //     assemble  : 대류 + 변수-μ 확산의 명시 RHS 조립
    //     buoyancy  : 부력항 gz·(1/ρ̂)·T̂ 추가 (T는 W 위치로 면-보간; Report/34 수정)
    //     pressure_gradient : 증분투영 예측자 항 −dt·(1/ρ̂)∇Pⁿ 추가 (누적압력 P_acc 사용)
    //     solve     : 3-sweep ADI(TDMA) + 블록 하삼각 속도 커플링으로 증분 δU,δV,δW 계산
    //     update    : U += δU 등으로 중간 속도 완성 (아직 발산 ≠ 0)
    equation::momentum::assemble_momentum_variable_property_gpu(domain, gpu_fields, gpu_properties.Mu, gpu_properties.invRho, gpu_momentum, dt,
                                   rec, timing_assemble_base);
    physics::buoyancy::add_buoyancy_force_variable_property_gpu(domain, gpu_momentum, gpu_fields[core::Var::T], gpu_properties.invRho, buoyancy_params, dt);
    main_mark(kMainTimingBuoyancy);
    equation::pressure::add_pressure_gradient_variable_property_gpu(domain, gpu_momentum, gpu_fields[core::Var::P], gpu_properties.invRho, dt);
    main_mark(kMainTimingPressureGradient);
    equation::momentum::solve_momentum_variable_property_gpu(domain, bc, gpu_fields, gpu_properties.Mu, gpu_properties.invRho, gpu_momentum, dt,
                                   rec, timing_momentum_base);
    equation::momentum::update_velocity_gpu(gpu_fields, gpu_momentum);
    main_mark(kMainTimingVelocityUpdate);
    core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);   // 할로 교환 + BC 재적용 (U,V,W)
    core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
    cudaDeviceSynchronize();
    // 디버그: 운동량 직후 상단 슬랩 |U|,|W| 최대값 (환경변수 MPMSTD_MOMDBG≥1일 때만 출력, 평소 no-op)
    equation::momentum::debug_slabmax_gpu("U.postmom", gpu_fields[core::Var::U], domain, 'U');
    equation::momentum::debug_slabmax_gpu("W.postmom", gpu_fields[core::Var::W], domain, 'W');

    // [6] 벽면 W=0 강제 — MAC 격자에서 z-벽의 W 면(내부 인덱스 k=h)은 TDMA에 안 접혀서
    //     솔브 후 명시적으로 0으로 눌러줘야 함
    // zero_w_wall: GPU kernel — no D2H/H2D
    cudaDeviceSynchronize();
    {
      const auto nt = sub.n_total();
      const int n1t = nt[0], n2t = nt[1], n3t = nt[2], h = kHaloWidth;
      const bool has_lower = (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
      const bool has_upper = (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
      const dim3 blk(16, 16);
      const dim3 grd((n1t+15)/16, (n2t+15)/16);
      k_zero_w_wall_gpu<<<grd, blk>>>(gpu_fields[core::Var::W].data(), n2t, n3t, h, n1t, has_lower, has_upper);
    }
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
    cudaDeviceSynchronize();
    main_mark(kMainTimingVelocityBoundary);

    // [7] 압력 투영(incremental van Kan) — ∇·U*를 RHS로 Poisson을 풀어(δp)
    //     펜슬 FFT(x,y) + 분산 TDMA(z), 속도를 발산-제로로 투영, P_acc += δp
    // G4: GPU pressure solve (NOB — no H↔D transfers for pressure)
    // Report/25: faithful NOB pressure restored. ExtraTerm (Poisson) and the
    // (invRhoc-1)*∇dPhat velocity term are a cancelling PAIR (div=0 exact), not
    // double-counting; the 2026-07-02 OB swap was a misdiagnosis (θ_c unchanged).
    equation::pressure::solve_pressure_variable_property_gpu(domain, bc, gpu_fields, pressure, dt, gpu_properties.invRho,
                                   rec, timing_pressure_base);
    equation::pressure::accumulate_pressure_gpu(gpu_fields, pressure);  // Pⁿ⁺¹ = Pⁿ + δp
    cudaDeviceSynchronize();
    main_mark(kMainTimingPressureUpdate);
    equation::momentum::debug_slabmax_gpu("U.postproj", gpu_fields[core::Var::U], domain);
    equation::momentum::debug_slabmax_gpu("W.postproj", gpu_fields[core::Var::W], domain);

    // [8] 온도 클리핑(NOB 전용, DeltaT>0일 때만) — 물성 다항식의 유효 범위를 벗어나는
    //     오버슛 T를 잘라냄; 몇 셀이 잘렸는지 d_clip_count에 기록해 모니터 때 보고
    // T clipping: GPU kernel — no D2H/H2D (interior only; ghosts hold the BC mirror)
    if (DeltaT > 0.0) {
      const auto nic = sub.n_interior();
      const auto ntc = sub.n_total();
      const dim3 blkc(8, 8, 8);
      const dim3 grdc((nic[0]+7)/8, (nic[1]+7)/8, (nic[2]+7)/8);
      k_clip_T_gpu<<<grdc, blkc>>>(gpu_fields[core::Var::T].data(),
                                   nic[0], nic[1], nic[2], ntc[1], ntc[2],
                                   statistics.clip_counter());
    }
    cudaDeviceSynchronize();
    main_mark(kMainTimingTemperatureClip);

    time += double(dt); ++step;   // 시간 전진 — 여기까지가 물리 스텝, 이후는 통계/출력

    // [9a] 통계 누적 (물리시간 t >= t_stat_start 이후) — θ_c, Nu, z-프로파일 디바이스 리덕션
    // GPU statistics: device reductions only — no field D2H (≤48 bytes per step)
    statistics.accumulate_if_due(time, gpu_fields);
    main_mark(kMainTimingStatistics);

    // [9b] 모니터 출력 (nmonitor 스텝마다) — step/t/dt/div/θ_c/Nu 한 줄 + monitor.csv + clip 보고
    statistics.monitor_if_due(step, time, dt, gpu_fields);
    main_mark(kMainTimingStatistics);
    if (rec) {
      rec->end_iteration();
      if (timing_iteration - timing_warmup == timing_measure - 1) {
        timing_recorder->stop_window();
        timing_window_done = true;
      }
      rec = nullptr;
    }
    ++timing_iteration;

    // [9c] 통계 파일 저장 (물리시간 t_stat_out 간격마다) — stats_t<time>.dat
    //      각 파일 = t_stat_start부터의 누적평균 (창이 커지는 cumulative average)
    statistics.write_profile_if_due(time, step);

    // [9d] 주기적 재시작 체크포인트 — 필드를 D2H 복사 후 ckpt_a/ckpt_b 교대로 기록
    //      (중간에 잡이 죽어도 최소 한 벌은 온전; P는 누적압력 P_acc를 저장)
    // Periodic restart checkpoint into alternating ckpt_a/ckpt_b (see the
    // nrestart_write comment at config parse). Skips the final step — the
    // end-of-run write below covers it in <dir_out> itself.
    if (restart.checkpoint_due(step) && step < n_steps && time < t_end) {
      core::copy_all_to_host(gpu_fields, cpu_fields);   // U/V/W/T/P device→host
      post::RestartState rs; rs.step = step; rs.time = time;
      rs.extras = statistics.export_extras();
      restart.checkpoint_if_due(step, time, cpu_fields, sub, rs, topo.cart_comm());
    }

  }

  // ── scaling-run timing epilogue: reduce + summary + one CSV (spec 8/10/11) ──
  if (timing_recorder && timing_window_done) {
    timing_recorder->reduce();
    std::vector<std::pair<std::string, std::vector<int>>> timing_groups;
    { std::vector<int> members{kMainTimingScalarHalo};
      for (int e = 0; e < equation::scalar::kGpuTimingEventCount; ++e) members.push_back(timing_scalar_base + e);
      timing_groups.emplace_back("scalar_total", members); }
    { std::vector<int> members{kMainTimingBuoyancy, kMainTimingPressureGradient,
                               kMainTimingVelocityUpdate, kMainTimingVelocityBoundary};
      for (int e = 0; e < equation::momentum::kGpuAssembleTimingEventCount; ++e) members.push_back(timing_assemble_base + e);
      for (int e = 0; e < equation::momentum::kGpuSolveTimingEventCount; ++e) members.push_back(timing_momentum_base + e);
      timing_groups.emplace_back("momentum_total", members); }
    { std::vector<int> members{kMainTimingPressureUpdate};
      for (int e = 0; e < equation::pressure::kGpuPressureTimingEventCount; ++e) members.push_back(timing_pressure_base + e);
      timing_groups.emplace_back("pressure_total", members); }
    timing_groups.emplace_back("tdma_z_all", std::vector<int>{
        timing_scalar_base   + equation::scalar::kTimingTdmaZ,
        timing_momentum_base + equation::momentum::kSolveTimingTdmaZ,
        timing_pressure_base + equation::pressure::kPressureTimingTdmaZ});
    timing_groups.emplace_back("halo_all", std::vector<int>{
        kMainTimingScalarHalo, kMainTimingVelocityBoundary,
        timing_momentum_base + equation::momentum::kSolveTimingHalo,
        timing_pressure_base + equation::pressure::kPressureTimingHalo});
    timing_recorder->print_summary(timing_groups);

    if (timing_dir != "none") {
      if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(timing_dir, ec); }
      MPI_Barrier(topo.cart_comm());
      int timing_world = 1; MPI_Comm_size(topo.cart_comm(), &timing_world);
      const char* job_id = std::getenv("SLURM_JOB_ID");
      const std::string grid_tag = std::to_string(n_global[0]) + "x" +
          std::to_string(n_global[1]) + "x" + std::to_string(n_global[2]);
      const std::string csv_path = timing_dir + "/timing_rbc_" + grid_tag + "_" +
          std::to_string(timing_world) + "gpu_run" + (job_id ? job_id : "local") + ".csv";
      std::vector<std::pair<std::string, std::string>> timing_metadata = {
        {"case", "rbc_gpu_nob"},
        {"global_grid", grid_tag},
        {"process_grid", std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x" + std::to_string(dims[2])},
        {"ngpu", std::to_string(timing_world)},
        {"Ra", std::to_string(Ra)}, {"Pr", std::to_string(Pr)}, {"DeltaT", std::to_string(DeltaT)},
        {"MaxCFL", std::to_string(double(max_cfl))}, {"dt_cap", std::to_string(double(dt_cap))},
        {"n_warmup_iteration", std::to_string(timing_warmup)},
        {"n_measured_iteration", std::to_string(timing_measure)},
        {"event_semantics", "partition events tile each step; total derived per rank (apps/rbc/timing_solve_plan.md)"},
      };
      timing_recorder->write_csv(csv_path, timing_metadata);
    }
  } else if (timing_recorder && mpi.is_root()) {
    std::printf("[timing] run ended before the measurement window completed - no timing output\n");
  }

  // Final stats file (stats_final.dat), end-of-run z-profile D2H + summary print.
  statistics.finalize(step);

  // Write restart files (U/V/W/T/P + stats.txt) for continuation runs
  if (!restart_out.empty()) {
    core::copy_all_to_host(gpu_fields, cpu_fields);   // U/V/W/T/P device→host
    post::RestartState rs; rs.step = step; rs.time = time;
    rs.extras = statistics.export_extras();
    restart.save(cpu_fields, sub, rs, topo.cart_comm());
  }
  return 0;
}
