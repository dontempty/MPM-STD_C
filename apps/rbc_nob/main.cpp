// MPM-STD RBC NOB 드라이버 — 가변 물성(글리세롤) Rayleigh–Bénard 대류.
//
// 수치·루프 순서는 MPM-STD_C apps/rbc/main_cpu_nob.cpp 1:1:
//   [CFL dt] → [물성 갱신(Tⁿ)] → [에너지: 가변-κ̂ 융합 ADI → Tⁿ⁺¹]
//   → [운동량 rhs(가변 μ̂) + 부력(Tⁿ⁺¹, 1/ρ̂·ρ 2차항) − (1/ρ̂)∇Pⁿ]
//   → [가변 운동량 solve: 전진 dU→dV→dW + 사후 커플링] → [동기화]
//   → [증분 투영 + ExtraTerm(p̂) + 가변 보정] → [P 누적·p̂ 외삽] → [T 클립(내부만)]
// OB(rbc 앱)와의 차이: 증분 투영(P 누적·p̂ 예측자), 물성 필드 6장, 전진 커플링.
// 상수물성 극한(glycerol(ΔT=0))은 전부 ≡1 로 환원된다 — test_rbc_nob 가 검증.
//
// 검증 목표(fig9): θ_c·ΔT vs Horn/Zhang/PaScaL_TCS — ΔT=40 에서 참조 재현값
// θ_c = 0.1383 (MPM-STD_C P7, Horn +0.05%).
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
#include "equation/momentum/rhs/rhs_variable.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/rhs/rhs.hpp"
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
  mpmstd::boundary::fill_z_wall_mirror(T, sub, kWallTemperatureBottom,
                                       kWallTemperatureTop);
}

// IC — 참조 main_cpu_nob 1:1 (rbc 앱과 동일): 전도 선형 T(z) + 다중모드 W 섭동.
void conduction_initial_condition(Field& W, Field& T, const mpmstd::grid::Grid& grid,
                                  const Subdomain& sub, real_t lx, real_t ly,
                                  real_t lz, real_t perturb) {
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
            kWallTemperatureBottom
            + (kWallTemperatureTop - kWallTemperatureBottom) * (z / lz);
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
    // 노드 안의 랭크 번호로 GPU 를 나눠 갖는다 — 모든 cudaMalloc 보다 먼저.
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
      throw std::runtime_error("rbc_nob: gravity_axis must be 2 (z)");
    if (s.physics.delta_temperature <= 0)
      throw std::runtime_error(
          "rbc_nob: delta_temperature must be > 0 (dimensional K) — the constant-"
          "property OB case is the rbc app");
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
    std::vector<real_t> z_center(s.mesh.nz), dz_global(s.mesh.nz);
    for (int q = 0; q < s.mesh.nz; ++q) {
      z_center[q] = axis_z_global.center[q + halo_width];
      dz_global[q] = axis_z_global.cell_width[q + halo_width];
    }

    // 자유낙하 무차원 물성 기준값 + 글리세롤 모델 (ΔT 는 차원 K — 다항식 인자)
    const real_t Cmu = std::sqrt(s.physics.prandtl / s.physics.rayleigh);
    const real_t Ct = 1.0 / std::sqrt(s.physics.rayleigh * s.physics.prandtl);
    const physics::PropertyModel model =
        physics::PropertyModel::glycerol(s.physics.delta_temperature);
    const real_t density_quadratic_ratio = model.density_quadratic_ratio();

    // 변수 — U, V, W, T, 누적 P + δp, 압력 예측자 p̂(dPhat)·p̂₀(dP0), rhs 3장,
    // 그리고 NOB 물성 6장 (이 앱만 할당 — OB 앱은 물성 필드가 없다).
    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"),
        T(nx, ny, nz, "T"), P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"),
        rhs_W(nx, ny, nz, "rhs_W"), rhs_dP(nx, ny, nz, "rhs_dP"),
        dP(nx, ny, nz, "dP");
    Field pressure_predictor(nx, ny, nz, "dPhat"),
        previous_predictor(nx, ny, nz, "dP0");
    physics::PropertyFields properties(nx, ny, nz);
    momentum::Solver momentum_solver(sub);
    scalar::Solver scalar_solver(sub);
    pressure::Solver pressure_solver(grid, sub);

    // ── 냉시작 vs 재시작 (M11) — 상태장 7장: U V W T P p̂ p̂₀.
    //    δp 는 매 스텝 fill(0) 후 재계산이라 상태가 아니다. ghost 는 저장하지
    //    않으므로 아래 sync 들이 IC 때와 똑같이 재구축한다.
    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.restart_from.empty()) {
      mpmstd::post::read_checkpoint(
          s.initial.restart_from, restart_step, restart_time,
          {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T}, {"P", &P},
           {"dPhat", &pressure_predictor}, {"dP0", &previous_predictor}},
          {}, sub);
      if (mpi.is_root())
        std::printf("# restart: %s (step=%ld, t=%.4f)\n",
                    s.initial.restart_from.c_str(), restart_step, restart_time);
    } else {
      conduction_initial_condition(W, T, grid, sub, s.domain.lx, s.domain.ly,
                                   s.domain.lz, s.initial.perturb);
    }
    sync_velocity(U, V, W, sub);
    sync_temperature(T, sub);
    sub.exchange_halo(P.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(P, sub);
    sub.exchange_halo(pressure_predictor.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(pressure_predictor, sub);
    sub.exchange_halo(previous_predictor.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(previous_predictor, sub);
    if (mpi.is_root() && s.output.checkpoint_interval > 0)
      std::filesystem::create_directories(s.output.checkpoint_dir);

    if (mpi.is_root()) {
      std::printf(
          "# NOB glycerol: Ra=%.3g Pr=%.3g DeltaT=%.1fK -> Cmu=%.4e Ct=%.4e "
          "a2/a1=%.4e | %dx%dx%d, dz_wall=%.4g\n",
          s.physics.rayleigh, s.physics.prandtl, s.physics.delta_temperature,
          Cmu, Ct, density_quadratic_ratio, s.mesh.nx, s.mesh.ny, s.mesh.nz,
          faces_z[1] - faces_z[0]);
      std::printf("# step      time        dt     max|div|   theta_c      Nu      "
                  "Tmin     Tmax   sec/step\n");
    }

    // ── 시간 평균 통계 (OB rbc 앱과 동일 형식 + ΔT 헤더) ──
    std::vector<double> sum_t_z(s.mesh.nz, 0.0), sum_w2_z(s.mesh.nz, 0.0),
        sum_wt_z(s.mesh.nz, 0.0);
    double sum_wt_integral = 0.0, sum_theta_c = 0.0;
    long samples = 0;
    const int kc_low = s.mesh.nz / 2 - 1, kc_high = s.mesh.nz / 2;

    const auto running_nusselt = [&]() {
      // 무차원 ΔT = 1 (벽 ±0.5) — delta_temperature 는 물성 다항식의 차원 인자다
      return (samples > 0) ? 1.0 + (sum_wt_integral / samples) / Ct : 0.0;
    };
    long stats_written_step = -1;
    const auto write_statistics = [&](long step_now, double t_now) {
      if (!mpi.is_root() || samples == 0 || step_now == stats_written_step) return;
      stats_written_step = step_now;
      char path[512];
      std::snprintf(path, sizeof(path), "%s/rbc_nob_stats_%08ld.dat",
                    s.output.stats_dir.c_str(), step_now);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      std::fprintf(f, "# rbc_nob statistics  step=%ld t=%.4f samples=%ld\n", step_now,
                   t_now, samples);
      std::fprintf(f,
                   "# Ra=%g Pr=%g DeltaT=%g Nu_volume=%.4f theta_c=%.5f\n",
                   s.physics.rayleigh, s.physics.prandtl,
                   s.physics.delta_temperature, running_nusselt(),
                   sum_theta_c / samples);
      std::fprintf(f, "# z  T_mean  w_rms  wT_mean\n");
      for (int q = 0; q < s.mesh.nz; ++q)
        std::fprintf(f, "%12.8f %12.8f %12.8f %14.8e\n", z_center[q],
                     sum_t_z[q] / samples,
                     std::sqrt(std::max(0.0, sum_w2_z[q] / samples)),
                     sum_wt_z[q] / samples);
      std::fclose(f);
      std::printf("# stats written: %s  (theta_c=%.5f, Nu=%.4f)\n", path,
                  sum_theta_c / samples, running_nusselt());
    };

    real_t t = static_cast<real_t>(restart_time);
    long step = restart_step;
    double next_stats_write = s.output.stats_start_time + s.output.stats_interval;
    if (s.output.stats_interval > 0)
      while (next_stats_write <= t) next_stats_write += s.output.stats_interval;
    // 재시작 시 통계 누적은 새로 시작한다 — 구간별 누적 스냅샷을 빼기 병합하는
    // 기존 워크플로 전제 (validation_time_mean 과 동일).
    std::vector<std::string> checkpoint_history;
    const auto write_restart_checkpoint = [&](long step_now, double t_now) {
      if (s.output.checkpoint_interval <= 0) return;
      char ck_path[512];
      std::snprintf(ck_path, sizeof(ck_path), "%s/rbc_nob_ck_%08ld.bin",
                    s.output.checkpoint_dir.c_str(), step_now);
      mpmstd::post::write_checkpoint(
          ck_path, step_now, t_now,
          {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T}, {"P", &P},
           {"dPhat", &pressure_predictor}, {"dP0", &previous_predictor}},
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
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max);

      // ── 물성 갱신 (Tⁿ — ghost 포함; 참조 루프 자리) ──
      physics::update_properties(properties, T, model, sub);

      // ── 에너지 (uⁿ, 가변-κ̂ 융합 ADI) → Tⁿ⁺¹ ──
      scalar::solve_variable_property(scalar_solver, grid, sub, T, U, V, W, properties, Ct, dt);
      sync_temperature(T, sub);

      // ── 운동량 예측자 (가변 μ̂; 부력은 Tⁿ⁺¹, 예측자는 −(1/ρ̂)∇Pⁿ) ──
      rhs_U.fill(0);
      rhs_V.fill(0);
      rhs_W.fill(0);
      momentum::add_advection_diffusion_u_variable_property(rhs_U, U, V, W,
                                                   properties.viscosity,
                                                   properties.inverse_density, grid,
                                                   sub, Cmu, dt);
      momentum::add_advection_diffusion_v_variable_property(rhs_V, U, V, W,
                                                   properties.viscosity,
                                                   properties.inverse_density, grid,
                                                   sub, Cmu, dt);
      momentum::add_advection_diffusion_w_variable_property(rhs_W, U, V, W,
                                                   properties.viscosity,
                                                   properties.inverse_density, grid,
                                                   sub, Cmu, dt);
      physics::add_buoyancy_z_variable(rhs_W, T, properties.inverse_density, grid,
                                       sub, density_quadratic_ratio,
                                       s.physics.delta_temperature, dt);
      pressure::add_gradient_variable(rhs_U, rhs_V, rhs_W, P,
                                      properties.inverse_density, grid, sub, dt);
      momentum::solve_variable_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V,
                               W, properties.viscosity, properties.inverse_density,
                               Cmu, dt);
      sync_velocity(U, V, W, sub);

      // ── 증분 투영 + NOB ExtraTerm/보정 (상수계수 Poisson 은 그대로) ──
      // ⚠ 알려진 보류 수정: 원본은 여기서 물성을 Tⁿ⁺¹ 로 재갱신한다 —
      //    docs/Debug/20261005_rbc_nob_projection_properties.patch (사용자
      //    결정으로 캠페인 일관성 위해 보류, A/B 검증은 완료).
      rhs_dP.fill(0);
      pressure::add_divergence(rhs_dP, U, V, W, grid, sub, dt);
      pressure::add_extra_term_variable(rhs_dP, pressure_predictor,
                                        properties.inverse_density, grid, sub);
      pressure::solve(pressure_solver, grid, sub, rhs_dP, dP);
      pressure::correct_velocity_variable(U, V, W, dP, pressure_predictor,
                                          properties.inverse_density, grid, sub, dt);
      pressure::accumulate(P, dP);  // 증분: Pⁿ⁺¹ = Pⁿ + δp (ghost 포함)
      sync_velocity(U, V, W, sub);

      // 다음 스텝용 p̂ 2차 외삽 + ghost 동기화 (참조 §820-821 + ghostcell_update)
      pressure::advance_pressure_predictor(pressure_predictor, previous_predictor, dP);
      sub.exchange_halo(pressure_predictor.data());
      mpmstd::boundary::fill_z_wall_zero_gradient(pressure_predictor, sub);

      // ── T 물리 클립 — **내부만** (ghost 까지 클램프하면 mirror 벽이 깨진다) ──
      mpmstd::equation::clip_interior(T, sub, kWallTemperatureTop,
                                      kWallTemperatureBottom);

      t += dt;
      ++step;
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_restart_checkpoint(step, t);

      // ── 통계 (물리시간 기준 시작, 매 스텝 표본) ──
      if (t >= s.output.stats_start_time) {
        const auto t_bar = mpmstd::post::plane_average(T, sub);
        const auto w2 = mpmstd::post::plane_average_square(W, sub);
        const auto wt = mpmstd::post::plane_average_product(W, T, sub);
        double wt_integral = 0.0;
        for (int q = 0; q < s.mesh.nz; ++q) {
          sum_t_z[q] += t_bar[q];
          sum_w2_z[q] += w2[q];
          sum_wt_z[q] += wt[q];
          wt_integral += wt[q] * dz_global[q];
        }
        sum_wt_integral += wt_integral;
        sum_theta_c += 0.5 * (t_bar[kc_low] + t_bar[kc_high]);
        ++samples;
        if (s.output.stats_interval > 0 && t >= next_stats_write) {
          write_statistics(step, t);
          next_stats_write += s.output.stats_interval;
        }
      }

      if (step % s.output.monitor_interval == 0 || step == 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        const auto t_bar = mpmstd::post::plane_average(T, sub);
        const double theta_c_now = 0.5 * (t_bar[kc_low] + t_bar[kc_high]);
        double t_min = t_bar[0], t_max = t_bar[0];
        for (int q = 0; q < s.mesh.nz; ++q) {
          t_min = std::min(t_min, double(t_bar[q]));
          t_max = std::max(t_max, double(t_bar[q]));
        }
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %9.5f %8.4f %8.4f %8.4f %8.3f\n",
                      step, t, dt, div, theta_c_now, running_nusselt(), t_min, t_max,
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
      write_restart_checkpoint(step, t);  // 연장 런을 위한 마지막 상태
    if (mpi.is_root() && samples > 0)
      std::printf("# FINAL: theta_c = %.5f (x DeltaT = %.3fK), Nu = %.4f  "
                  "(t=%.1f, samples=%ld)\n",
                  sum_theta_c / samples,
                  sum_theta_c / samples * s.physics.delta_temperature,
                  running_nusselt(), t, samples);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    return 1;
  }
  return 0;
}
