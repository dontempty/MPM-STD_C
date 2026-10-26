// ══ 시간 측정용 쌍둥이 main — tools/scale/instrument_main.py 가 main.cpp 에서 자동 생성 (손으로 고치지 말 것).
//    make <app>_scale 가 -DMPMSTD_SCALE_MAIN 으로 컴파일한다 (일반 빌드에서는 빈 TU). 구간: bulk_correct, dt_cfl, momentum_rhs, momentum_solve, monitor, pressure_correct, pressure_rhs, pressure_solve, scalar_solve, statistics, sync_T, sync_UVW.
#ifdef MPMSTD_SCALE_MAIN
// MPM-STD DHVC 드라이버 — 차등가열 연직공동(Differentially Heated Vertical
// Cavity), 상수물성(OB) 공기(Pr=0.7). MPM-STD_C apps/dhvc/main_gpu.cu 1:1.
//
// 축 관례: x = 연직·중력·주류 (주기, 부력이 U 에 실림),  y = 스팬 (주기),
//          z = 벽수직 (hot z=0: T=+0.5, cold z=Lz: T=−0.5).
// 무차원: ν = √(Pr/Ra),  α = 1/√(Ra·Pr).  부력계수 1 (자유낙하 스케일).
//
// 한 스텝 (참조 recipe 1:1):
//   [CFL dt] → [에너지: 상수-α 융합 ADI → Tⁿ⁺¹]
//   → [운동량 rhs(상수 ν) + 부력_x(Tⁿ⁺¹) − ∇Pⁿ] → [상수 운동량 solve]
//   → [순 x유량 0 보정] → [동기화] → [증분 투영] → [P 누적] → [통계(Re_δ*)]
//
// 통계(2026-10-07 확장, apps/dhvc/statistics.hpp): 평면 프로파일(U V W T·rms·⟨uw⟩⟨uT⟩⟨wT⟩·벽 Nu·u_τ·θ_τ·
//   Re_δ*·Re_rms·⟨dP/dx⟩) + 국소 |Ri_f| 조건평균·면적비·PDF + u 스펙트럼 — Ng et al. (2017) JFM 825 전 항목.
//   모니터에 순간 Nu·u_τ. 표본 간격은 [output] stats_sample_interval / conditional_interval / spectra_interval.
//
// ★ 상수물성 전용 경로 — 물성 배열(μ̂·1/ρ̂·κ̂)을 **만들지도 참조하지도 않는다**
//   (*_constant_property; 사용자 지시 2026-10-05). 가변물성(NOB)은 rbc_nob.
//
// 검증 목표(fig7): Re_δ* vs PaScaL_TCS — Ra 1e5/1e6/1e7/1e8.
//   Re_δ* = u_max · δ* / ν,  δ* = ∫₀^{z(u_max)} (1 − U/u_max) dz,  U = ⟨u⟩_{xy,t}.
#include <mpi.h>
#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "boundary/wall.hpp"
#include "equation/momentum/rhs/rhs.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/correction.hpp"
#include "equation/pressure/rhs/rhs.hpp"
#include "equation/pressure/solve.hpp"
#include "equation/scalar/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"
#include "physics/buoyancy.hpp"
#include "physics/channel_forcing.hpp"
#include "post/checkpoint.hpp"
#include <tuple>
#include "post/statistics.hpp"
#include "dhvc/statistics.hpp"
#include "rbc/settings.hpp"
#include "time/time_step.hpp"
#include <cstdlib>
#include "common/phase_timer.hpp"

using mpmstd::halo_width;
using mpmstd::real_t;
using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;
using mpmstd::parallel::Topology;
namespace momentum = mpmstd::equation::momentum;
namespace pressure = mpmstd::equation::pressure;
namespace scalar = mpmstd::equation::scalar;
namespace physics = mpmstd::physics;

namespace {

constexpr real_t kWallTemperatureHot = 0.5;    // z = 0  (뜨거운 벽)
constexpr real_t kWallTemperatureCold = -0.5;  // z = Lz (차가운 벽)

// 접선 속도 벽 ghost: value(1차, 참조 PaScaL/MPM-STD_C 관례 — 기본) | mirror(2차, 벽면 no-slip;
// 운동량 닫기 B−=A 와 짝). [boundary] velocity_ghost 로 고른다 (채널과 동일, 2026-10-07).
void sync_velocity(Field& U, Field& V, Field& W, const Subdomain& sub, const bool mirror) {
  sub.exchange_halo(U.data());
  sub.exchange_halo(V.data());
  sub.exchange_halo(W.data());
  if (mirror) {
    mpmstd::boundary::fill_z_wall_mirror(U, sub, 0.0, 0.0);
    mpmstd::boundary::fill_z_wall_mirror(V, sub, 0.0, 0.0);
  } else {
    mpmstd::boundary::fill_z_wall_value(U, sub, 0.0, 0.0);
    mpmstd::boundary::fill_z_wall_value(V, sub, 0.0, 0.0);
  }
  mpmstd::boundary::zero_z_wall_faces(W, sub);
}

void sync_temperature(Field& T, const Subdomain& sub) {
  sub.exchange_halo(T.data());
  mpmstd::boundary::fill_z_wall_mirror(T, sub, kWallTemperatureHot,
                                       kWallTemperatureCold);
}

// 냉시작 IC — 참조 main_gpu.cu 1:1: 포물선 U(z) base + 선형 T(z) + 다중모드 섭동.
void cold_start_initial_condition(Field& U, Field& V, Field& W, Field& T,
                                  const mpmstd::grid::Grid& grid, const Subdomain& sub,
                                  real_t lx, real_t ly, real_t lz, real_t nu,
                                  real_t perturb) {
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const auto at = [&](int i, int j, int k) {
    return (static_cast<std::size_t>(i) * ny + j) * nz + k;
  };
  std::vector<real_t> su(U.size(), 0.0), sv(V.size(), 0.0), sw(W.size(), 0.0),
      st(T.size(), 0.0);
  const real_t pi = std::acos(real_t(-1));
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        const real_t x = grid.x.center[i], y = grid.y.center[j], z = grid.z.center[k];
        su[at(i, j, k)] = z * (2 * z / lz - 1) * (z / lz - 1) / (12 * nu);
        st[at(i, j, k)] =
            kWallTemperatureHot + (kWallTemperatureCold - kWallTemperatureHot) * (z / lz);
        const real_t envelope = std::sin(pi * z / lz);
        sv[at(i, j, k)] =
            perturb * envelope
            * (std::sin(2 * pi * x / lx) * std::cos(2 * pi * y / ly)
               + std::sin(4 * pi * x / lx) * std::sin(2 * pi * y / ly));
        sw[at(i, j, k)] =
            real_t(0.5) * perturb * envelope * std::sin(2 * pi * x / lx);
      }
  U.upload(su);
  V.upload(sv);
  W.upload(sw);
  T.upload(st);
}

}  // namespace

int main(int argc, char** argv) {
  mpmstd::parallel::MpiContext mpi(&argc, &argv);
#ifdef MPMSTD_GPU
  {
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices > 0) cudaSetDevice(mpi.node_rank() % n_devices);
  }
#endif
  if (argc < 2) {
    if (mpi.is_root()) std::fprintf(stderr, "usage: %s <input.ini>\n", argv[0]);
    return 1;
  }
  try {
    const rbc::Settings s = rbc::load_settings(argv[1]);
    const bool mirror_wall =
        s.boundary.condition.velocity_ghost == mpmstd::boundary::VelocityGhost::Mirror;
    if (s.physics.gravity_axis != 0)
      throw std::runtime_error("dhvc: gravity_axis must be 0 (x, 연직)");
    if (s.physics.delta_temperature != 0)
      throw std::runtime_error(
          "dhvc: delta_temperature must be 0 (상수물성 OB) — NOB 는 rbc_nob");

    if (mpi.is_root()) {
      std::printf("# settings loaded from %s  (%d ranks)\n", argv[1], mpi.world_size());
      std::fputs(rbc::format_settings(s).c_str(), stdout);
      std::filesystem::create_directories(s.output.stats_dir);
    }

    const Topology topo(mpi, s.process.npx, s.process.npy, s.process.npz,
                        /*periodic_x=*/true, /*periodic_y=*/true, /*periodic_z=*/false);
    const Subdomain sub(topo, s.mesh.nx, s.mesh.ny, s.mesh.nz);
    const auto faces_x = mpmstd::grid::make_uniform_faces(s.mesh.nx, s.domain.lx);
    const auto faces_y = mpmstd::grid::make_uniform_faces(s.mesh.ny, s.domain.ly);
    const auto faces_z =
        s.stretch.uniform_z
            ? mpmstd::grid::make_uniform_faces(s.mesh.nz, s.domain.lz)
            : mpmstd::grid::make_tanh_faces(s.mesh.nz, s.domain.lz, s.stretch.gamma_z);
    const mpmstd::grid::Grid grid{
        mpmstd::grid::make_axis(faces_x, {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
        mpmstd::grid::make_axis(faces_y, {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
        mpmstd::grid::make_axis(faces_z, {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
    };
    const auto axis_z_global =
        mpmstd::grid::make_axis(faces_z, {s.mesh.nz, 0, s.mesh.nz}, "z");
    std::vector<real_t> z_center(s.mesh.nz);
    for (int q = 0; q < s.mesh.nz; ++q) z_center[q] = axis_z_global.center[q + halo_width];

    const real_t nu = std::sqrt(s.physics.prandtl / s.physics.rayleigh);
    const real_t Ct = real_t(1) / std::sqrt(s.physics.rayleigh * s.physics.prandtl);

    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"),
        T(nx, ny, nz, "T"), P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"), rhs_W(nx, ny, nz, "rhs_W");
    // [메모리 2026-10-26] 압력 우변·보정은 운동량 우변 메모리를 재사용한다 (docs/Memory/plan.md §9).
    //   한 스텝에서 rhs_U 의 역할: 운동량 우변 → (momentum::solve 안에서) δU → 압력 우변 → (pressure::solve 는 제자리) dP.
    //   운동량 → 압력 순서가 고정이라 두 역할이 동시에 살아 있는 순간이 없다. 되돌리려면 아래 두 참조를 Field 생성으로.
    Field& rhs_P = rhs_U;
    Field& dP = rhs_P;      // pressure::solve 가 rhs 자리에 해를 쓴다
  // [최적화 2026-10-23] 솔버 공용 작업 영역 — 세 솔버가 밴드·스펙트럼 버퍼를 빌려 쓴다 (docs/Memory/plan.md)
    mpmstd::equation::Workspace workspace(std::max({momentum::Solver::required_slot_elements(sub),
                                                    scalar::Solver::required_slot_elements(sub),
                                                    pressure::Solver::required_slot_elements(grid, sub)}));
    momentum::Solver momentum_solver(sub, workspace);
    scalar::Solver scalar_solver(sub, workspace);
    if (mirror_wall) {
      // ghost 정책과 일치하는 음해 닫기: 속도 거울 → B−=A, 온도(거울 ghost) → B−=A.
      // value(기본)는 drop(참조 1:1)을 유지한다.
      momentum::set_wall_closure(momentum_solver, s.boundary.condition);
      scalar::set_wall_closure(scalar_solver, s.boundary.condition, mpmstd::boundary::Field::T);
    }
    pressure::Solver pressure_solver(grid, sub, workspace);
    real_t mean_dPdx = 0;  // 순 x유량 0 보정의 적립(진단) — 재시작 상태

    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.warmstart_from.empty()) {
      // 격자 교차 warm-start — 하위 Ra 의 restart 를 이 격자에 trilinear 보간.
      using mpmstd::post::Stagger;
      mpmstd::post::read_checkpoint_interpolated(
          s.initial.warmstart_from, restart_step, restart_time,
          {std::make_tuple(std::string("U"), &U, Stagger::FaceX),
           std::make_tuple(std::string("V"), &V, Stagger::FaceY),
           std::make_tuple(std::string("W"), &W, Stagger::FaceZ),
           std::make_tuple(std::string("T"), &T, Stagger::Center),
           std::make_tuple(std::string("P"), &P, Stagger::Center)},
          {{"mean_dPdx", &mean_dPdx}}, grid, sub, s.domain.lx, s.domain.ly,
          s.domain.lz, s.stretch.uniform_z, s.stretch.gamma_z);
      restart_step = 0;     // 새 Ra 케이스 — 스텝·시간·통계 전부 0 부터 (참조 t_start=0;
      restart_time = 0;     // 보간장은 초기조건일 뿐, 시간 적분은 새로 시작)
      if (mpi.is_root())
        std::printf("# warm-start(격자보간): %s (t=%.4f)\n",
                    s.initial.warmstart_from.c_str(), restart_time);
    } else if (!s.initial.restart_from.empty()) {
      mpmstd::post::read_checkpoint(s.initial.restart_from, restart_step, restart_time,
                                    {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T},
                                     {"P", &P}},
                                    {{"mean_dPdx", &mean_dPdx}}, sub);
      if (mpi.is_root())
        std::printf("# restart: %s (step=%ld, t=%.4f)\n",
                    s.initial.restart_from.c_str(), restart_step, restart_time);
    } else {
      cold_start_initial_condition(U, V, W, T, grid, sub, s.domain.lx, s.domain.ly,
                                   s.domain.lz, nu, s.initial.perturb);
    }
    sync_velocity(U, V, W, sub, mirror_wall);
    sync_temperature(T, sub);
    sub.exchange_halo(P.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(P, sub);
    if (mpi.is_root() && s.output.checkpoint_interval > 0)
      std::filesystem::create_directories(s.output.checkpoint_dir);

    // 투영으로 IC 발산 1회 청소 (참조 methodology step 4)
    rhs_P.fill(0);
    pressure::add_divergence(rhs_P, U, V, W, grid, sub, real_t(1));
    pressure::solve(pressure_solver, grid, sub, rhs_P, dP);
    pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, real_t(1));
    pressure::accumulate(P, dP);
    sync_velocity(U, V, W, sub, mirror_wall);

    if (mpi.is_root()) {
      std::printf("# DHVC OB: Ra=%.3g Pr=%.3g -> nu=%.6e alpha=%.6e | %dx%dx%d\n",
                  s.physics.rayleigh, s.physics.prandtl, nu, Ct, s.mesh.nx, s.mesh.ny,
                  s.mesh.nz);
      std::printf("# step      time        dt     max|div|    Re_dstar    u_max   "
                  "Nu_hot   Nu_cold  u_tau_hot  u_tau_cold  sec/step\n");
    }

    // ── 통계 (apps/dhvc/statistics.hpp): 평면 프로파일 + |Ri_f| 조건평균 + u 스펙트럼 — Ng 2017 전 항목 ──
    dhvc::StatisticsConfig stats_config;
    stats_config.nz = s.mesh.nz;
    stats_config.lx = s.domain.lx;
    stats_config.ly = s.domain.ly;
    stats_config.lz = s.domain.lz;
    stats_config.rayleigh = s.physics.rayleigh;
    stats_config.prandtl = s.physics.prandtl;
    stats_config.nu = nu;
    stats_config.alpha = Ct;
    stats_config.mirror_wall = mirror_wall;
    stats_config.z_center = z_center;
    stats_config.ghost_gap_hot = z_center[0] - axis_z_global.center[halo_width - 1];
    stats_config.ghost_gap_cold =
        axis_z_global.center[s.mesh.nz + halo_width] - z_center[s.mesh.nz - 1];
    stats_config.wall_temperature_hot = kWallTemperatureHot;
    stats_config.wall_temperature_cold = kWallTemperatureCold;
    stats_config.sample_interval = s.output.stats_sample_interval;
    stats_config.conditional_interval = s.output.conditional_interval;
    stats_config.spectra_interval = s.output.spectra_interval;
    stats_config.richardson_threshold = s.output.richardson_threshold;
    stats_config.richardson_z_plus = s.output.richardson_z_plus;
    stats_config.stats_dir = s.output.stats_dir;
    dhvc::Statistics statistics(stats_config, sub);
    const auto running_re_delta = [&]() -> real_t { return statistics.running_re_delta(); };

    long stats_written_step = -1;
    const auto write_statistics = [&](long step_now, double t_now) {
      if (!statistics.started() || step_now == stats_written_step) return;
      stats_written_step = step_now;
      statistics.write(step_now, t_now, mpi.is_root());  // 루트만 파일, 전 랭크 호출(결과는 복제돼 있음)
    };

    const auto write_checkpoint = [&](long step_now, double t_now) {
      if (s.output.checkpoint_interval <= 0) return;
      char ck[512];
      std::snprintf(ck, sizeof(ck), "%s/dhvc_ck_%08ld.bin",
                    s.output.checkpoint_dir.c_str(), step_now);
      mpmstd::post::write_checkpoint(ck, step_now, t_now,
                                     {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T},
                                      {"P", &P}},
                                     {{"mean_dPdx", mean_dPdx}}, sub);
      if (mpi.is_root()) std::printf("# checkpoint written: %s (t=%.4f)\n", ck, t_now);
    };

    real_t t = static_cast<real_t>(restart_time);
    long step = restart_step;
    double next_stats_write =
        s.output.stats_interval > 0 ? s.output.stats_start_time + s.output.stats_interval
                                    : 0;
    while (next_stats_write > 0 && next_stats_write <= t)
      next_stats_write += s.output.stats_interval;
    double wall_prev = MPI_Wtime();

    // [scale] 구간 타이머 — tools/scale/instrument_main.py 가 main.cpp 에서 생성 (수치 불변, 동기화만 추가)
    mpmstd::common::PhaseTimer timer;
    const long timing_start = std::getenv("MPMSTD_PHASE_START") ? std::atol(std::getenv("MPMSTD_PHASE_START")) : 50;
    if (timing_start > 0) { momentum_solver.timer = &timer; pressure_solver.timer = &timer; }
    if (timing_start > 0) scalar_solver.timer = &timer;
    double tdma_base_momentum = 0, tdma_base_pressure = 0, tdma_base_scalar = 0;
    while (t < s.time.end_time && (s.time.max_step == 0 || step < s.time.max_step)) {
      if (timing_start > 0 && !timer.active() && step >= timing_start) {
        timer.activate();
        tdma_base_momentum = momentum_solver.plan_z.comm_seconds;
        tdma_base_pressure = pressure_solver.plan_z.comm_seconds;
        tdma_base_scalar = scalar_solver.plan_z.comm_seconds;
      }
      timer.begin("step_total");
      timer.begin("dt_cfl");
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max); timer.end("dt_cfl");

      // ── 에너지 (uⁿ, 상수-α 융합 ADI) → Tⁿ⁺¹ ──
      { timer.begin("scalar_solve");
      scalar::solve_constant_property(scalar_solver, grid, sub, T, U, V, W, Ct, dt); timer.end("scalar_solve"); }
      { timer.begin("sync_T");
      sync_temperature(T, sub); timer.end("sync_T"); }

      // ── 운동량 예측자 (상수 ν) + 부력_x(Tⁿ⁺¹) − ∇Pⁿ ──
      // [최적화 2026-10-18] 전용 RHS 빌드 — fill·이송확산 ×3·부력 x·∇P 를 성분당 한 패스로 (순서·식 동일)
      { timer.begin("momentum_rhs");
      momentum::build_constant_property_rhs(rhs_U, rhs_V, rhs_W, U, V, W, P, grid, sub, nu, dt, &T); timer.end("momentum_rhs"); }
      { timer.begin("momentum_solve");
      momentum::solve_constant_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W,
                                        U, V, W, nu, dt); timer.end("momentum_solve"); }
      { timer.begin("sync_UVW");
      sync_velocity(U, V, W, sub, mirror_wall); timer.end("sync_UVW"); }

      // ── 순 x유량 0 (연직 가속 억제; 참조 apply_mass_flow_correction target 0) ──
      const real_t mean_dPdx_before = mean_dPdx;
      { timer.begin("bulk_correct");
      physics::correct_bulk_velocity(U, grid, sub, /*target_bulk=*/0, dt, mean_dPdx); timer.end("bulk_correct"); }
      // 이 스텝에 실제로 가한 등가 ∂P/∂x (shift = −dt·increment). DHVC 는 누적 mean_dPdx 를
      // 힘으로 다시 쓰지 않으므로(채널의 add_mean_pressure_gradient 없음) 스텝별 적용분이 수지의 힘.
      const real_t dpdx_applied = mean_dPdx - mean_dPdx_before;
      { timer.begin("sync_UVW");
      sync_velocity(U, V, W, sub, mirror_wall); timer.end("sync_UVW"); }

      // ── 증분 투영 ──
      { timer.begin("pressure_rhs");
      rhs_P.fill(0); timer.end("pressure_rhs"); }
      { timer.begin("pressure_rhs");
      pressure::add_divergence(rhs_P, U, V, W, grid, sub, dt); timer.end("pressure_rhs"); }
      { timer.begin("pressure_solve");
      pressure::solve(pressure_solver, grid, sub, rhs_P, dP); timer.end("pressure_solve"); }
      { timer.begin("pressure_correct");
      pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, dt); timer.end("pressure_correct"); }
      { timer.begin("pressure_correct");
      pressure::accumulate(P, dP); timer.end("pressure_correct"); }
      { timer.begin("sync_UVW");
      sync_velocity(U, V, W, sub, mirror_wall); timer.end("sync_UVW"); }

      t += dt;
      ++step;
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_checkpoint(step, t);

      // ── 통계 (물리시간 ≥ stats_start; 표본 간격·dt 가중은 Statistics 가 관리) ──
      if (t >= s.output.stats_start_time) {
        { timer.begin("statistics");
        statistics.accumulate(step, t, dt, dpdx_applied, U, V, W, T, grid, sub); timer.end("statistics"); }
        if (s.output.stats_interval > 0 && t >= next_stats_write) {
          write_statistics(step, t);
          next_stats_write += s.output.stats_interval;
        }
      }

      if (step % s.output.monitor_interval == 0 || step == 1) {
        timer.begin("monitor");
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub); timer.end("monitor");
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        const auto t_bar = mpmstd::post::plane_average(T, sub);
        real_t u_max = 0;
        for (int q = 0; q < s.mesh.nz; ++q) u_max = std::max(u_max, u_bar[q]);
        double nu_hot = 0, nu_cold = 0, ut_hot = 0, ut_cold = 0;  // 순간값 (면평균)
        statistics.wall_nusselt(t_bar, nu_hot, nu_cold);
        statistics.wall_friction_velocity(u_bar, ut_hot, ut_cold);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %11.4f %9.5f %8.3f %8.3f %10.6f %10.6f %8.3f\n",
                      step, t, dt, div, running_re_delta(), u_max, nu_hot, nu_cold, ut_hot,
                      ut_cold, (wall_now - wall_prev) / s.output.monitor_interval);
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        if (!(div == div) || div > 1e3)
          throw std::runtime_error("divergence blew up — aborting");
      }
      timer.end("step_total"); timer.count_step();
    }
    if (timing_start > 0) {
      timer.set_external("tdma_comm_momentum_z", momentum_solver.plan_z.comm_seconds - tdma_base_momentum);
      timer.set_external("tdma_comm_pressure_z", pressure_solver.plan_z.comm_seconds - tdma_base_pressure);
      timer.set_external("tdma_comm_scalar_z", scalar_solver.plan_z.comm_seconds - tdma_base_scalar);
      int world_ranks = 1; MPI_Comm_size(MPI_COMM_WORLD, &world_ranks);
      const char* csv = std::getenv("MPMSTD_PHASE_CSV");
      timer.report(MPI_COMM_WORLD, csv ? csv : "phase_timing.csv",
                   {{"nx", std::to_string(s.mesh.nx)}, {"ny", std::to_string(s.mesh.ny)}, {"nz", std::to_string(s.mesh.nz)},
                    {"npz", std::to_string(s.process.npz)}, {"ranks", std::to_string(world_ranks)},
                    {"phase_timing_start_step", std::to_string(timing_start)}});
    }

    write_statistics(step, t);
    if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval != 0)
      write_checkpoint(step, t);
    if (mpi.is_root())
      std::printf("# FINAL: Re_dstar = %.4f  (Ra = %g, t = %.1f)\n", running_re_delta(),
                  s.physics.rayleigh, t);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return 0;
}
#endif  // MPMSTD_SCALE_MAIN
