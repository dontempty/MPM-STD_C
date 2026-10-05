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

constexpr real_t kWallTemperatureHot = 0.5;    // z = 0  (뜨거운 벽)
constexpr real_t kWallTemperatureCold = -0.5;  // z = Lz (차가운 벽)

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

// Re_δ* = u_max·δ*/ν  (참조 compute_re_delta_star 1:1). U_profile = ⟨u⟩_{xy,t}.
real_t reynolds_delta_star(const std::vector<double>& u_profile,
                           const std::vector<real_t>& z_center, const real_t nu) {
  const int nz = static_cast<int>(u_profile.size());
  int kmax = 0;
  double u_max = u_profile[0];
  for (int k = 0; k < nz; ++k)
    if (u_profile[k] > u_max) {
      u_max = u_profile[k];
      kmax = k;
    }
  if (u_max <= 0.0) return 0.0;
  double dstar = 0.0, z_prev = 0.0, f_prev = 1.0;  // 벽 z=0, U=0 → 1−0 = 1
  for (int k = 0; k <= kmax; ++k) {
    const double z = z_center[k];
    const double f = 1.0 - u_profile[k] / u_max;
    dstar += 0.5 * (f_prev + f) * (z - z_prev);
    z_prev = z;
    f_prev = f;
  }
  return static_cast<real_t>(u_max * dstar / nu);
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
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"),
        rhs_W(nx, ny, nz, "rhs_W"), rhs_P(nx, ny, nz, "rhs_P"), dP(nx, ny, nz, "dP");
    momentum::Solver momentum_solver(sub);
    scalar::Solver scalar_solver(sub);
    pressure::Solver pressure_solver(grid, sub);
    real_t mean_dPdx = 0;  // 순 x유량 0 보정의 적립(진단) — 재시작 상태

    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.restart_from.empty()) {
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
    sync_velocity(U, V, W, sub);
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
    sync_velocity(U, V, W, sub);

    if (mpi.is_root()) {
      std::printf("# DHVC OB: Ra=%.3g Pr=%.3g -> nu=%.6e alpha=%.6e | %dx%dx%d\n",
                  s.physics.rayleigh, s.physics.prandtl, nu, Ct, s.mesh.nx, s.mesh.ny,
                  s.mesh.nz);
      std::printf("# step      time        dt     max|div|    Re_dstar    u_max   "
                  "sec/step\n");
    }

    // 시간(dt 가중) 누적 U(z) 프로파일 → Re_δ*
    std::vector<double> sum_u_profile(s.mesh.nz, 0.0);
    double sum_dt = 0.0;
    const auto running_re_delta = [&]() -> real_t {
      if (sum_dt <= 0.0) return 0;
      std::vector<double> mean(s.mesh.nz);
      for (int q = 0; q < s.mesh.nz; ++q) mean[q] = sum_u_profile[q] / sum_dt;
      return reynolds_delta_star(mean, z_center, nu);
    };

    long stats_written_step = -1;
    const auto write_statistics = [&](long step_now, double t_now) {
      if (!mpi.is_root() || sum_dt <= 0.0 || step_now == stats_written_step) return;
      stats_written_step = step_now;
      char path[512];
      std::snprintf(path, sizeof(path), "%s/dhvc_stats_%08ld.dat",
                    s.output.stats_dir.c_str(), step_now);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      std::fprintf(f, "# dhvc statistics  step=%ld t=%.4f\n", step_now, t_now);
      std::fprintf(f, "# Ra=%g Pr=%g Re_dstar=%.6f\n", s.physics.rayleigh,
                   s.physics.prandtl, running_re_delta());
      std::fprintf(f, "# z  U_mean\n");
      for (int q = 0; q < s.mesh.nz; ++q)
        std::fprintf(f, "%12.8f %14.8e\n", z_center[q], sum_u_profile[q] / sum_dt);
      std::fclose(f);
      std::printf("# stats written: %s  (Re_dstar=%.4f)\n", path, running_re_delta());
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

    while (t < s.time.end_time && (s.time.max_step == 0 || step < s.time.max_step)) {
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max);

      // ── 에너지 (uⁿ, 상수-α 융합 ADI) → Tⁿ⁺¹ ──
      scalar::solve_constant_property(scalar_solver, grid, sub, T, U, V, W, Ct, dt);
      sync_temperature(T, sub);

      // ── 운동량 예측자 (상수 ν) + 부력_x(Tⁿ⁺¹) − ∇Pⁿ ──
      rhs_U.fill(0);
      rhs_V.fill(0);
      rhs_W.fill(0);
      momentum::add_advection_diffusion_u_constant_property(rhs_U, U, V, W, grid, sub, nu, dt);
      momentum::add_advection_diffusion_v_constant_property(rhs_V, U, V, W, grid, sub, nu, dt);
      momentum::add_advection_diffusion_w_constant_property(rhs_W, U, V, W, grid, sub, nu, dt);
      physics::add_buoyancy_x_constant_property(rhs_U, T, grid, sub, dt);
      pressure::add_gradient_constant_property(rhs_U, rhs_V, rhs_W, P, grid, sub, dt);
      momentum::solve_constant_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W,
                                        U, V, W, nu, dt);
      sync_velocity(U, V, W, sub);

      // ── 순 x유량 0 (연직 가속 억제; 참조 apply_mass_flow_correction target 0) ──
      physics::correct_bulk_velocity(U, grid, sub, /*target_bulk=*/0, dt, mean_dPdx);
      sync_velocity(U, V, W, sub);

      // ── 증분 투영 ──
      rhs_P.fill(0);
      pressure::add_divergence(rhs_P, U, V, W, grid, sub, dt);
      pressure::solve(pressure_solver, grid, sub, rhs_P, dP);
      pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, dt);
      pressure::accumulate(P, dP);
      sync_velocity(U, V, W, sub);

      t += dt;
      ++step;
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_checkpoint(step, t);

      // ── 통계 (물리시간 ≥ stats_start, dt 가중, 매 스텝) ──
      if (t >= s.output.stats_start_time) {
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        for (int q = 0; q < s.mesh.nz; ++q) sum_u_profile[q] += double(u_bar[q]) * dt;
        sum_dt += dt;
        if (s.output.stats_interval > 0 && t >= next_stats_write) {
          write_statistics(step, t);
          next_stats_write += s.output.stats_interval;
        }
      }

      if (step % s.output.monitor_interval == 0 || step == 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        real_t u_max = 0;
        for (int q = 0; q < s.mesh.nz; ++q) u_max = std::max(u_max, u_bar[q]);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %11.4f %9.5f %8.3f\n", step, t, dt, div,
                      running_re_delta(), u_max,
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
