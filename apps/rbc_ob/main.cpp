// MPM-STD RBC OB 드라이버 — **충실 상수물성 경로** (2026-10-13 신설).
//
// apps/rbc 는 legacy_ob 간이 스킴(보존형 RHS + 짝 없는 이류형 음해항; 운동량이 dt 에
// 비례해 샘 — equation/momentum/legacy_ob/rhs.hpp 머리말)이라 Pr 2547·dt 0.05 에서 벽이
// 미끄럼처럼 거동했다(벽 인접 u_rms 가 NOB 의 6배, w_rms ∝ z, Nu 14 vs 9.5; 2026-10-13 진단).
// 이 앱은 DHVC 가 검증한 충실 커널을 쓴다:
//   momentum::add_advection_diffusion_{u,v,w}_constant_property + solve_constant_property
//   scalar::solve_constant_property (융합 ADI), 증분 투영(P 누적, ∇Pⁿ 예측자)
// 루프 순서·IC·벽 ghost 정책(U/V zero-ghost, W 벽면 0, T mirror)·T 클립·통계(Horn 2013 전
// 항목, rbc_nob/statistics 공유 — 물성 필드는 ≡1)·체크포인트는 rbc_nob 과 1:1 이라
// NOB(ΔT>0) 과 OB(이 앱) 의 차이는 물성 모델뿐이다. 설정은 rbc 와 같은 키.
//
// 목적(첫 사용): NOB 윗벽(차가운 벽) 속도 rms 스파이크가 상수물성에서 사라지는지 — 같은
// 격자·Ra·Pr·CFL·통계 프로토콜로 윗/아랫벽 근처 rms 대칭 확인.
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
#include "equation/pressure/rhs/rhs.hpp"
#include "equation/pressure/correction.hpp"
#include "equation/pressure/solve.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/sweep_kernels.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"
#include "physics/buoyancy.hpp"
#include "physics/properties.hpp"
#include "post/checkpoint.hpp"
#include "post/statistics.hpp"
#include "rbc/settings.hpp"
#include "rbc_nob/statistics.hpp"
#include "time/time_step.hpp"

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

constexpr real_t kWallTemperatureBottom = 0.5;  // 뜨거운 바닥 (무차원; ΔT_무차원 = 1)
constexpr real_t kWallTemperatureTop = -0.5;    // 차가운 천장

// rbc_nob 과 동일: U/V 는 zero-ghost(ghost = 0), W 는 벽면 0
void sync_velocity(Field& U, Field& V, Field& W, const Subdomain& sub) {
  sub.exchange_halo(U.data());
  mpmstd::boundary::fill_z_wall_value(U, sub, 0.0, 0.0);
  sub.exchange_halo(V.data());
  mpmstd::boundary::fill_z_wall_value(V, sub, 0.0, 0.0);
  sub.exchange_halo(W.data());
  mpmstd::boundary::zero_z_wall_faces(W, sub);
}

void sync_temperature(Field& T, const Subdomain& sub) {
  sub.exchange_halo(T.data());
  mpmstd::boundary::fill_z_wall_mirror(T, sub, kWallTemperatureBottom, kWallTemperatureTop);
}

// IC — rbc/rbc_nob 1:1: 전도 선형 T(z) + 다중모드 W 섭동
void conduction_initial_condition(Field& W, Field& T, const mpmstd::grid::Grid& grid,
                                  const Subdomain& sub, real_t lx, real_t ly, real_t lz,
                                  real_t perturb) {
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const auto at = [&](int i, int j, int k) {
    return (static_cast<std::size_t>(i) * ny + j) * nz + k;
  };
  std::vector<real_t> stage_w(W.size(), 0.0), stage_t(T.size(), 0.0);
  const real_t pi = std::acos(real_t(-1));
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        const real_t x = grid.x.center[i], y = grid.y.center[j], z = grid.z.center[k];
        stage_t[at(i, j, k)] =
            kWallTemperatureBottom + (kWallTemperatureTop - kWallTemperatureBottom) * (z / lz);
        const real_t envelope = std::sin(pi * z / lz);
        stage_w[at(i, j, k)] =
            perturb * envelope
            * (std::sin(2 * pi * x / lx) * std::cos(2 * pi * y / ly)
               + std::cos(2 * pi * x / lx) * std::sin(2 * pi * y / ly)
               + std::sin(4 * pi * x / lx) * std::sin(4 * pi * y / ly));
      }
  W.upload(stage_w);
  T.upload(stage_t);
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
    if (s.physics.gravity_axis != 2)
      throw std::runtime_error("rbc_ob: gravity_axis must be 2 (z)");
    if (s.physics.delta_temperature != 0)
      throw std::runtime_error("rbc_ob: delta_temperature must be 0 (constant properties) — "
                               "the NOB glycerol case is the rbc_nob app");
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
    const auto faces_z = s.stretch.uniform_z
                             ? mpmstd::grid::make_uniform_faces(s.mesh.nz, s.domain.lz)
                             : mpmstd::grid::make_tanh_faces(s.mesh.nz, s.domain.lz,
                                                             s.stretch.gamma_z);
    const mpmstd::grid::Grid grid{
        mpmstd::grid::make_axis(faces_x, {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
        mpmstd::grid::make_axis(faces_y, {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
        mpmstd::grid::make_axis(faces_z, {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
    };
    const auto axis_z_global = mpmstd::grid::make_axis(faces_z, {s.mesh.nz, 0, s.mesh.nz}, "z");
    std::vector<real_t> z_center(s.mesh.nz), dz_global(s.mesh.nz);
    for (int q = 0; q < s.mesh.nz; ++q) {
      z_center[q] = axis_z_global.center[q + halo_width];
      dz_global[q] = axis_z_global.cell_width[q + halo_width];
    }

    // 자유낙하 무차원 상수 물성: ν = √(Pr/Ra), α = 1/√(Ra·Pr)
    const real_t Cmu = std::sqrt(s.physics.prandtl / s.physics.rayleigh);
    const real_t Ct = 1.0 / std::sqrt(s.physics.rayleigh * s.physics.prandtl);

    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"), T(nx, ny, nz, "T"),
        P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"), rhs_W(nx, ny, nz, "rhs_W"),
        rhs_dP(nx, ny, nz, "rhs_dP"), dP(nx, ny, nz, "dP");
    // 통계 모듈(rbc_nob/statistics)이 ⟨μ̂⟩⟨κ̂⟩·Re_E 에 물성 필드를 읽는다 — 상수물성이라 ≡1 로 한 번 채운다.
    physics::PropertyFields properties(nx, ny, nz);
    const physics::PropertyModel unit_model;  // 선행 계수만 1 → 모든 물성 ≡ 1, 미분 ≡ 0
    momentum::Solver momentum_solver(sub);
    scalar::Solver scalar_solver(sub);
    pressure::Solver pressure_solver(grid, sub);

    // ── 냉시작 vs 재시작 — 상태장 5장 U V W T P (ghost 는 아래 sync 가 재구축) ──
    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.restart_from.empty()) {
      mpmstd::post::read_checkpoint(s.initial.restart_from, restart_step, restart_time,
                                    {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T}, {"P", &P}}, {},
                                    sub);
      if (mpi.is_root())
        std::printf("# restart: %s (step=%ld, t=%.4f)\n", s.initial.restart_from.c_str(),
                    restart_step, restart_time);
    } else {
      conduction_initial_condition(W, T, grid, sub, s.domain.lx, s.domain.ly, s.domain.lz,
                                   s.initial.perturb);
    }
    sync_velocity(U, V, W, sub);
    sync_temperature(T, sub);
    sub.exchange_halo(P.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(P, sub);
    physics::update_properties(properties, T, unit_model, sub);
    if (mpi.is_root() && s.output.checkpoint_interval > 0)
      std::filesystem::create_directories(s.output.checkpoint_dir);

    if (restart_step == 0) {
      // 투영으로 IC 발산 1회 청소 (DHVC 와 동일; 섭동 W 는 발산이 0 이 아니다)
      rhs_dP.fill(0);
      pressure::add_divergence(rhs_dP, U, V, W, grid, sub, real_t(1));
      pressure::solve(pressure_solver, grid, sub, rhs_dP, dP);
      pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, real_t(1));
      pressure::accumulate(P, dP);
      sync_velocity(U, V, W, sub);
    }

    if (mpi.is_root()) {
      std::printf("# OB constant property (faithful kernels): Ra=%.3g Pr=%.3g -> nu=%.4e alpha=%.4e "
                  "| %dx%dx%d, dz_wall=%.4g\n",
                  s.physics.rayleigh, s.physics.prandtl, Cmu, Ct, s.mesh.nx, s.mesh.ny,
                  s.mesh.nz, faces_z[1] - faces_z[0]);
      std::printf("# step      time        dt     max|div|   theta_c      Nu      "
                  "Tmin     Tmax    Nu_b     Nu_t   u_bot_max  u_top_max  sec/step\n");
    }

    // ── 통계 (rbc_nob/statistics: Horn 2013 전 항목; ΔT = 0, 물성 ≡ 1) ──
    rbc_nob::StatisticsConfig stats_config;
    stats_config.nz = s.mesh.nz;
    stats_config.lx = s.domain.lx;
    stats_config.ly = s.domain.ly;
    stats_config.lz = s.domain.lz;
    stats_config.rayleigh = s.physics.rayleigh;
    stats_config.prandtl = s.physics.prandtl;
    stats_config.delta_temperature = 0;
    stats_config.Cmu = Cmu;
    stats_config.Ct = Ct;
    stats_config.model = unit_model;
    stats_config.z_center = z_center;
    stats_config.dz = dz_global;
    stats_config.sample_interval = s.output.stats_sample_interval;
    stats_config.stats_dir = s.output.stats_dir;
    rbc_nob::Statistics statistics(stats_config, sub);
    const auto running_nusselt = [&]() { return statistics.nusselt_volume(); };
    long stats_written_step = -1;
    const auto write_statistics = [&](long step_now, double t_now) {
      if (!statistics.started() || step_now == stats_written_step) return;
      stats_written_step = step_now;
      statistics.write(step_now, t_now, mpi.is_root());
    };

    real_t t = static_cast<real_t>(restart_time);
    long step = restart_step;
    double next_stats_write = s.output.stats_start_time + s.output.stats_interval;
    if (s.output.stats_interval > 0)
      while (next_stats_write <= t) next_stats_write += s.output.stats_interval;
    std::vector<std::string> checkpoint_history;
    const auto write_restart_checkpoint = [&](long step_now, double t_now) {
      if (s.output.checkpoint_interval <= 0) return;
      char ck_path[512];
      std::snprintf(ck_path, sizeof(ck_path), "%s/rbc_ob_ck_%08ld.bin",
                    s.output.checkpoint_dir.c_str(), step_now);
      mpmstd::post::write_checkpoint(ck_path, step_now, t_now,
                                     {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T}, {"P", &P}},
                                     {}, sub);
      if (mpi.is_root()) {
        std::printf("# checkpoint written: %s (t=%.4f)\n", ck_path, t_now);
        std::fflush(stdout);
        checkpoint_history.push_back(ck_path);
        if (checkpoint_history.size() > 2) {  // 최신 2개만 유지
          std::remove(checkpoint_history.front().c_str());
          checkpoint_history.erase(checkpoint_history.begin());
        }
      }
    };

    double wall_prev = MPI_Wtime();
    while (t < s.time.end_time && (s.time.max_step == 0 || step < s.time.max_step)) {
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(U, V, W, grid, sub,
                                                                 s.time.max_cfl, s.time.dt_max);

      // ── 에너지 (uⁿ, 상수-α 융합 ADI) → Tⁿ⁺¹ ──
      scalar::solve_constant_property(scalar_solver, grid, sub, T, U, V, W, Ct, dt);
      sync_temperature(T, sub);

      // ── 운동량 예측자 (상수 ν) + 부력_z(Tⁿ⁺¹) − ∇Pⁿ → 완전결합 δ-form ADI ──
      rhs_U.fill(0);
      rhs_V.fill(0);
      rhs_W.fill(0);
      momentum::add_advection_diffusion_u_constant_property(rhs_U, U, V, W, grid, sub, Cmu, dt);
      momentum::add_advection_diffusion_v_constant_property(rhs_V, U, V, W, grid, sub, Cmu, dt);
      momentum::add_advection_diffusion_w_constant_property(rhs_W, U, V, W, grid, sub, Cmu, dt);
      physics::add_buoyancy_z_constant_property(rhs_W, T, grid, sub, dt);
      pressure::add_gradient_constant_property(rhs_U, rhs_V, rhs_W, P, grid, sub, dt);
      momentum::solve_constant_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V, W,
                                        Cmu, dt);
      sync_velocity(U, V, W, sub);

      // ── 증분 투영: Pⁿ⁺¹ = Pⁿ + δp ──
      rhs_dP.fill(0);
      pressure::add_divergence(rhs_dP, U, V, W, grid, sub, dt);
      pressure::solve(pressure_solver, grid, sub, rhs_dP, dP);
      pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, dt);
      pressure::accumulate(P, dP);
      sync_velocity(U, V, W, sub);

      // ── T 물리 클립 — 내부만 (rbc_nob 과 동일) ──
      mpmstd::equation::clip_interior(T, sub, kWallTemperatureTop, kWallTemperatureBottom);

      t += dt;
      ++step;
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_restart_checkpoint(step, t);

      if (t >= s.output.stats_start_time) {
        statistics.accumulate(step, t, dt, U, V, W, T, properties, grid, sub);
        if (s.output.stats_interval > 0 && t >= next_stats_write) {
          write_statistics(step, t);
          next_stats_write += s.output.stats_interval;
        }
      }

      if (step % s.output.monitor_interval == 0 || step == restart_step + 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        const auto t_bar = mpmstd::post::plane_average(T, sub);
        const double theta_c_now = statistics.center_temperature(t_bar);
        double t_min = t_bar[0], t_max = t_bar[0];
        for (int q = 0; q < s.mesh.nz; ++q) {
          t_min = std::min(t_min, double(t_bar[q]));
          t_max = std::max(t_max, double(t_bar[q]));
        }
        double nu_bottom_now = 0, nu_top_now = 0;
        statistics.wall_nusselt(t_bar, nu_bottom_now, nu_top_now);
        const auto u_abs_max = mpmstd::post::plane_maximum(U, sub, /*absolute=*/true);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %9.5f %8.4f %8.4f %8.4f %8.4f %8.4f %10.3e %10.3e %8.3f\n",
                      step, t, dt, div, theta_c_now, running_nusselt(), t_min, t_max,
                      nu_bottom_now, nu_top_now, double(u_abs_max[0]),
                      double(u_abs_max[s.mesh.nz - 1]),
                      (wall_now - wall_prev) / s.output.monitor_interval);
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        if (!(div == div) || div > 1e3)
          throw std::runtime_error("divergence blew up — aborting");
      }
    }

    write_statistics(step, t);
    if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval != 0)
      write_restart_checkpoint(step, t);
    if (mpi.is_root() && statistics.started())
      std::printf("# FINAL: theta_c = %.5f, Nu = %.4f  (t=%.1f, sum_dt=%.1f)\n",
                  statistics.theta_c(), running_nusselt(), t, statistics.sum_dt());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    return 1;
  }
  return 0;
}
