// MPM-STD RBC 드라이버 — Rayleigh–Bénard 대류 (x·y 주기, z 벽; OB 상수물성).
//
// 수치·루프 순서는 MPM-STD_C apps/rbc/main_cpu_ob.cpp 1:1:
//   [CFL dt] → [에너지: uⁿ 으로 rhs → CN-ADI → Tⁿ⁺¹] → [운동량 rhs + 부력(Tⁿ⁺¹)]
//   → [BW-ADI 운동량] → [투영] → [동기화] → [통계·모니터]
// 투영은 **비증분**(예측자에 ∇P 없음, 압력 누적 없음 — 참조 OB 경로 그대로).
// 그래서 이 앱에는 누적 압력 P 필드 자체가 없다 — δp 스크래치(dP)만 있다.
// 자유낙하 무차원: ν = √(Pr/Ra), α = 1/√(Ra·Pr), 부력 = +T (z), 벽 온도 ±0.5.
//
// 검증 지표 (참조 rbc_common 1:1):
//   Nu = 1 + time-mean[ ∫⟨wT⟩dz ] / (α·ΔT)   (슬롯값 w·T, dz 가중)
//   θ_c = 중앙 평면 ⟨T⟩ 의 시간 평균 (OB 대칭이면 → 0)
//   벽 Nu = (T_wall − ⟨T⟩₁)/zc₁ · lz/ΔT      (mirror 벽은 면 위가 정확 — 면 기준 자)
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
#include "equation/momentum/legacy_ob/rhs.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/rhs/rhs.hpp"
#include "equation/pressure/solve.hpp"
#include "equation/scalar/legacy_ob/rhs.hpp"
#include "equation/scalar/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"
#include "physics/buoyancy.hpp"
#include "post/statistics.hpp"
#include "settings.hpp"
#include "time/time_step.hpp"

using mpmstd::halo_width;
using mpmstd::real_t;
using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;
using mpmstd::parallel::Topology;
namespace momentum = mpmstd::equation::momentum;
namespace pressure = mpmstd::equation::pressure;
namespace scalar = mpmstd::equation::scalar;

namespace {

constexpr real_t kWallTemperatureBottom = 0.5;   // 뜨거운 바닥
constexpr real_t kWallTemperatureTop = -0.5;     // 차가운 천장 (ΔT = 1)

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

// IC — 참조 main_cpu_ob 1:1: 전도 선형 T(z) + 다중모드 W 섭동 (대류 세포 깨우기).
// GPU Field 는 호스트 원소 접근이 없으므로 호스트 스테이징에 만들어 올린다.
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
      throw std::runtime_error("rbc: gravity_axis must be 2 (z) — x-gravity cases "
                               "(DHVC) get add_buoyancy_x when they arrive");
    if (s.physics.delta_temperature != 0)
      throw std::runtime_error("rbc: delta_temperature > 0 is the NOB (glycerol) "
                               "path — not implemented yet (OB only)");
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
    // 전역 z 보조 (통계·Nu 용 — 호스트)
    const auto axis_z_global =
        mpmstd::grid::make_axis(faces_z, {s.mesh.nz, 0, s.mesh.nz}, "z");
    std::vector<real_t> z_center(s.mesh.nz), dz_global(s.mesh.nz);
    for (int q = 0; q < s.mesh.nz; ++q) {
      z_center[q] = axis_z_global.center[q + halo_width];
      dz_global[q] = axis_z_global.cell_width[q + halo_width];
    }

    // 자유낙하 무차원 물성 — 케이스 물리라 main 이 만든다.
    const real_t nu = std::sqrt(s.physics.prandtl / s.physics.rayleigh);
    const real_t alpha_T =
        1.0 / std::sqrt(s.physics.rayleigh * s.physics.prandtl);
    const real_t delta_temperature = kWallTemperatureBottom - kWallTemperatureTop;

    // 이 앱의 변수 — U, V, W, T (+ rhs 들과 δp 스크래치. 누적 P 없음 — 비증분).
    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"),
        T(nx, ny, nz, "T");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"),
        rhs_W(nx, ny, nz, "rhs_W"), rhs_T(nx, ny, nz, "rhs_T"),
        rhs_dP(nx, ny, nz, "rhs_dP"), dP(nx, ny, nz, "dP");
    momentum::Solver momentum_solver(sub);
    scalar::Solver scalar_solver(sub);
    pressure::Solver pressure_solver(grid, sub);

    conduction_initial_condition(W, T, grid, sub, s.domain.lx, s.domain.ly,
                                 s.domain.lz, s.initial.perturb);
    sync_velocity(U, V, W, sub);
    sync_temperature(T, sub);

    if (mpi.is_root()) {
      std::printf("# Ra=%.3g Pr=%.3g -> nu=%.4e alpha=%.4e | %dx%dx%d, dz_wall=%.4g\n",
                  s.physics.rayleigh, s.physics.prandtl, nu, alpha_T, s.mesh.nx,
                  s.mesh.ny, s.mesh.nz, faces_z[1] - faces_z[0]);
      std::printf("# step      time        dt     max|div|   theta_c      Nu    sec/step\n");
    }

    // ── 시간 평균 통계 (참조 RbcStats 1:1) ──
    std::vector<double> sum_t_z(s.mesh.nz, 0.0), sum_w2_z(s.mesh.nz, 0.0),
        sum_wt_z(s.mesh.nz, 0.0);
    double sum_wt_integral = 0.0, sum_theta_c = 0.0;
    long samples = 0;
    const int kc_low = s.mesh.nz / 2 - 1, kc_high = s.mesh.nz / 2;  // 중앙 두 평면 평균

    const auto running_nusselt = [&]() {
      return (samples > 0)
                 ? 1.0 + (sum_wt_integral / samples) / (alpha_T * delta_temperature)
                 : 0.0;
    };
    long stats_written_step = -1;  // 마지막 스텝에서 interval·종료 쓰기 중복 방지
    const auto write_statistics = [&](long step_now, double t_now) {
      if (!mpi.is_root() || samples == 0 || step_now == stats_written_step) return;
      stats_written_step = step_now;
      char path[512];
      std::snprintf(path, sizeof(path), "%s/rbc_stats_%08ld.dat",
                    s.output.stats_dir.c_str(), step_now);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      const double t_bar_1 = sum_t_z[0] / samples;
      const double t_bar_n = sum_t_z[s.mesh.nz - 1] / samples;
      const double nu_wall_bottom = (kWallTemperatureBottom - t_bar_1) / z_center[0]
                                    * s.domain.lz / delta_temperature;
      const double nu_wall_top = (t_bar_n - kWallTemperatureTop)
                                 / (s.domain.lz - z_center[s.mesh.nz - 1])
                                 * s.domain.lz / delta_temperature;
      std::fprintf(f, "# rbc statistics  step=%ld t=%.4f samples=%ld\n", step_now,
                   t_now, samples);
      std::fprintf(f, "# Ra=%g Pr=%g Nu_volume=%.4f Nu_wall_bottom=%.4f "
                   "Nu_wall_top=%.4f theta_c=%.5f\n",
                   s.physics.rayleigh, s.physics.prandtl, running_nusselt(),
                   nu_wall_bottom, nu_wall_top, sum_theta_c / samples);
      std::fprintf(f, "# z  T_mean  w_rms  wT_mean\n");
      for (int q = 0; q < s.mesh.nz; ++q)
        std::fprintf(f, "%12.8f %12.8f %12.8f %14.8e\n", z_center[q],
                     sum_t_z[q] / samples,
                     std::sqrt(std::max(0.0, sum_w2_z[q] / samples)),
                     sum_wt_z[q] / samples);
      std::fclose(f);
      std::printf("# stats written: %s  (Nu=%.4f)\n", path, running_nusselt());
    };

    real_t t = 0;
    long step = 0;
    double next_stats_write = s.output.stats_start_time + s.output.stats_interval;
    double wall_prev = MPI_Wtime();
    while (t < s.time.end_time
           && (s.time.max_step == 0 || step < s.time.max_step)) {
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max);

      // ── 에너지 (uⁿ 으로) → Tⁿ⁺¹ ──
      rhs_T.fill(0);
      scalar::add_diffusion(rhs_T, T, grid, sub, alpha_T, dt);
      scalar::add_convection(rhs_T, T, U, V, W, grid, sub, dt);
      scalar::solve(scalar_solver, grid, sub, rhs_T, T, alpha_T, dt);
      sync_temperature(T, sub);

      // ── 운동량 예측자 (+부력은 갱신된 Tⁿ⁺¹ — 참조 순서) ──
      rhs_U.fill(0);
      rhs_V.fill(0);
      rhs_W.fill(0);
      momentum::add_advection_diffusion_cell(rhs_U, U, U, V, W, grid, sub, nu, dt);
      momentum::add_advection_diffusion_cell(rhs_V, V, U, V, W, grid, sub, nu, dt);
      momentum::add_advection_diffusion_z_face(rhs_W, W, U, V, W, grid, sub, nu, dt);
      mpmstd::physics::add_buoyancy_z(rhs_W, T, grid, sub, dt);
      momentum::solve(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V, W, nu, dt);
      sync_velocity(U, V, W, sub);

      // ── 비증분 투영 (δp 만 — 누적 없음, 참조 OB 경로) ──
      rhs_dP.fill(0);
      pressure::add_divergence(rhs_dP, U, V, W, grid, sub, dt);
      pressure::solve(pressure_solver, grid, sub, rhs_dP, dP);
      pressure::correct_velocity(U, V, W, dP, grid, sub, dt);
      sync_velocity(U, V, W, sub);

      t += dt;
      ++step;

      // ── 통계 (물리시간 기준 시작, 매 스텝 표본 — 참조 1:1) ──
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
        if (t >= next_stats_write) {
          write_statistics(step, t);
          next_stats_write += s.output.stats_interval;
        }
      }

      if (step % s.output.monitor_interval == 0 || step == 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        const auto t_bar = mpmstd::post::plane_average(T, sub);
        const double theta_c_now = 0.5 * (t_bar[kc_low] + t_bar[kc_high]);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %9.5f %8.4f %8.4f\n", step, t, dt,
                      div, theta_c_now, running_nusselt(),
                      (wall_now - wall_prev) / s.output.monitor_interval);
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        if (!(div == div) || div > 1e3)
          throw std::runtime_error("divergence blew up — aborting");
      }
    }

    write_statistics(step, t);
    if (mpi.is_root() && samples > 0)
      std::printf("# FINAL: Nu = %.4f, theta_c = %.5f  (t=%.1f, samples=%ld)\n",
                  running_nusselt(), sum_theta_c / samples, t, samples);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    return 1;
  }
  return 0;
}
