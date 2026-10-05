// MPM-STD 채널 드라이버 — 난류 채널 (x 주류·y 스팬 주기, z 벽). 온도 없음.
//
// 문제 조건: Filtered_TDMA/apps/channel_gpu 의 Re_τ=180 케이스 1:1
//   (256³, 2π×π×2, Re_b = U_b·δ/ν = 2857, CFR 벌크 1 유지, tanh z,
//    IC = 포물선 + 반회전 종방향 와류쌍 + 잡음 — laminar_init(vortex) 이식).
// 한 스텝 = [CFL dt] → [운동량 rhs + ∇Pⁿ + 평균구배] → [BW-ADI 운동량] →
//           [CFR 보정] → [증분 투영] → [동기화] → [통계·모니터].
#include <mpi.h>
#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "boundary/wall.hpp"
#include "equation/momentum/rhs/rhs.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/rhs/rhs.hpp"
#include "equation/pressure/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"
#include "physics/channel_forcing.hpp"
#include "post/checkpoint.hpp"
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

namespace {

void sync_velocity(Field& U, Field& V, Field& W, const Subdomain& sub) {
  sub.exchange_halo(U.data());
  mpmstd::boundary::fill_z_wall_value(U, sub, 0.0, 0.0);
  sub.exchange_halo(V.data());
  mpmstd::boundary::fill_z_wall_value(V, sub, 0.0, 0.0);
  sub.exchange_halo(W.data());
  mpmstd::boundary::zero_z_wall_faces(W, sub);
}

// IC — channel_gpu laminar_init(init_mode = vortex) 1:1 이식.
// GPU 빌드에서 Field 는 호스트 원소 접근이 없으므로, IC 는 호스트 스테이징
// 배열에 만들어 upload() 로 한 번에 올린다 (양쪽 빌드 공통 코드).
// 포물선 U + 반회전 종방향 와류쌍(V, W) + 균일 잡음, 섭동 최대값을
// pert·U_max 로 스케일, 마지막에 벌크를 정확히 맞춘다.
void vortex_initial_condition(Field& U, Field& V, Field& W,
                              const mpmstd::grid::Grid& grid, const Subdomain& sub,
                              real_t ly, real_t lz, real_t bulk_velocity,
                              real_t pert, int world_rank) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const auto at = [&](int i, int j, int k) {
    return (static_cast<std::size_t>(i) * ny + j) * nz + k;
  };
  std::vector<real_t> stage_u(U.size(), 0.0), stage_v(U.size(), 0.0),
      stage_w(U.size(), 0.0);

  const real_t pi = std::acos(real_t(-1));
  const real_t u_max = real_t(1.5) * bulk_velocity;
  const real_t n_beta = std::max(real_t(1), std::round(ly / lz));
  const real_t beta = 2 * pi * n_beta / ly;
  const real_t kz = pi / lz;
  const real_t half = real_t(0.5) * lz;

  // 재현 가능한 잡음 (channel_gpu 와 같은 시드 규칙 — 랭크별 오프셋)
  std::mt19937 rng(1234567891ULL
                   ^ (static_cast<std::uint64_t>(world_rank) * 0x9E3779B97F4A7C15ULL));
  std::uniform_real_distribution<real_t> rnd(-0.5, 0.5);

  real_t local_max = 0;
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t y = grid.y.center[j];
        const real_t z = grid.z.center[k];
        const real_t s1 = std::sin(kz * z);
        stage_v[at(i, j, k)] = std::cos(beta * y) * kz * std::sin(2 * kz * z);
        stage_w[at(i, j, k)] = beta * std::sin(beta * y) * s1 * s1;
        local_max = std::max(local_max, std::max(std::abs(stage_v[at(i, j, k)]),
                                                 std::abs(stage_w[at(i, j, k)])));
      }
  real_t global_vortex_max = 0;
  MPI_Allreduce(&local_max, &global_vortex_max, 1, MPI_DOUBLE, MPI_MAX,
                sub.topology().cart_comm());
  const real_t scale = (global_vortex_max > 0) ? pert * u_max / global_vortex_max : 0;
  const real_t noise = real_t(0.1) * pert * u_max;

  // 체적 평균(슬롯값 그대로 — channel_gpu bulk 1:1)을 호스트에서 바로 계산
  real_t sums[4] = {0, 0, 0, 0};  // {∫u, ∫v, ∫w, ∫dV}
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t zr = (grid.z.center[k] - half) / half;
        const real_t u_laminar = u_max * std::max(real_t(0), 1 - zr * zr);
        stage_u[at(i, j, k)] = u_laminar + noise * rnd(rng);
        stage_v[at(i, j, k)] = scale * stage_v[at(i, j, k)] + noise * rnd(rng);
        stage_w[at(i, j, k)] = scale * stage_w[at(i, j, k)] + noise * rnd(rng);
        const real_t dv = grid.x.cell_width[i] * grid.y.cell_width[j]
                          * grid.z.cell_width[k];
        sums[0] += stage_u[at(i, j, k)] * dv;
        sums[1] += stage_v[at(i, j, k)] * dv;
        sums[2] += stage_w[at(i, j, k)] * dv;
        sums[3] += dv;
      }
  real_t totals[4] = {0, 0, 0, 0};
  MPI_Allreduce(sums, totals, 4, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  const real_t mean_u = totals[0] / totals[3];
  const real_t mean_v = totals[1] / totals[3];
  const real_t mean_w = totals[2] / totals[3];
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        stage_u[at(i, j, k)] += bulk_velocity - mean_u;  // 벌크(U) = U_b 정확히
        stage_v[at(i, j, k)] -= mean_v;
        stage_w[at(i, j, k)] -= mean_w;
      }

  U.upload(stage_u);
  V.upload(stage_v);
  W.upload(stage_w);
}

// 벽 마찰속도 — 스킴과 일관된 벽 전단으로 잰다: zero-ghost 벽에서 운동량이
// 보는 벽 전단은 (첫 중심값 − ghost 중심값 0) / **중심간거리 dmz** 다.
// ⚠ 벽면↔첫중심(= dmz/2)으로 재면 기울기가 2배 → u_τ 가 √2 과대 — 실측으로
//   Re_τ 258 vs 182 의 차이가 전부 이 자(길이) 선택이었다 (2026-10-04).
real_t friction_velocity(const std::vector<real_t>& u_profile, real_t dmz_bottom,
                         real_t dmz_top, real_t nu) {
  const real_t shear_bottom = u_profile.front() / dmz_bottom;
  const real_t shear_top = u_profile.back() / dmz_top;
  const real_t tau = real_t(0.5) * nu * (std::abs(shear_bottom) + std::abs(shear_top));
  return std::sqrt(tau);
}

}  // namespace

int main(int argc, char** argv) {
  mpmstd::parallel::MpiContext mpi(&argc, &argv);
#ifdef MPMSTD_GPU
  // 노드 안의 랭크 번호로 GPU 를 나눠 갖는다 (GPU 2장 = 랭크 2개).
  // MpiContext 의 node_rank 가 정확히 이 용도다 (M3). 모든 cudaMalloc 보다
  // 먼저 와야 하므로 필드 선언 전 — 여기가 유일한 자리다.
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
    const channel::Settings s = channel::load_settings(argv[1]);
    if (mpi.is_root()) {
      std::printf("# settings loaded from %s  (%d ranks)\n", argv[1], mpi.world_size());
      std::fputs(channel::format_settings(s).c_str(), stdout);
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
    // 전역 z 중심 좌표 (통계 출력·u_τ 용)
    const auto axis_z_global = mpmstd::grid::make_axis(
        faces_z, {s.mesh.nz, 0, s.mesh.nz}, "z");
    std::vector<real_t> z_center(s.mesh.nz);
    for (int q = 0; q < s.mesh.nz; ++q) z_center[q] = axis_z_global.center[q + halo_width];
    // zero-ghost 벽 전단의 자: ghost 중심 ↔ 첫/끝 내부 중심 거리 (= 벽 cell 폭)
    const real_t dmz_bottom = axis_z_global.center_distance[halo_width];
    const real_t dmz_top =
        axis_z_global.center_distance[axis_z_global.n_total - halo_width];

    const real_t delta = real_t(0.5) * s.domain.lz;  // 반높이 = 길이 척도
    const real_t bulk_target = s.physics.target_bulk_velocity;
    const real_t nu = bulk_target * delta / s.physics.reynolds_bulk;

    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"), P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"),
        rhs_W(nx, ny, nz, "rhs_W"), rhs_P(nx, ny, nz, "rhs_P"), dP(nx, ny, nz, "dP");
    momentum::Solver momentum_solver(sub);
    // ── 원본(core_momentum.f90) 충실 경로 = 상수물성 전용 커널 (plan §18-1) ──
    // OB 경로(보존형 RHS + 짝 없는 이류형 음해항)는 블록 커플링의 평균이 0 이
    // 아니어서 운동량이 dt 에 비례해 샌다 (CFR 이 벽 항력의 1.17~1.41 배로
    // 과구동 — 2026-10-04 수지 계측). 상수물성 충실 커널은 M-form 잔차·own ×2
    // 응력·cross-stress·전진 대입까지 참조 구조 그대로, **물성 배열을 만들지도
    // 참조하지도 않는다** — "가변 + μ̂≡1 배열" 과 비트 일치 검증됨.
    pressure::Solver pressure_solver(grid, sub);
    real_t mean_dPdx = 0;

    // ── 냉시작 vs 재시작 (M11) — 상태장 4장 + mean_dPdx. dP 는 매 스텝
    //    재계산이라 상태가 아니다. ghost 는 아래 sync 가 재구축한다.
    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.restart_from.empty()) {
      mpmstd::post::read_checkpoint(s.initial.restart_from, restart_step, restart_time,
                                    {{"U", &U}, {"V", &V}, {"W", &W}, {"P", &P}},
                                    {{"mean_dPdx", &mean_dPdx}}, sub);
      if (mpi.is_root())
        std::printf("# restart: %s (step=%ld, t=%.4f, mean_dPdx=%.6e)\n",
                    s.initial.restart_from.c_str(), restart_step, restart_time,
                    mean_dPdx);
    } else {
      vortex_initial_condition(U, V, W, grid, sub, s.domain.ly, s.domain.lz,
                               bulk_target, s.initial.pert_amp, mpi.world_rank());
    }
    sync_velocity(U, V, W, sub);
    sub.exchange_halo(P.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(P, sub);
    if (mpi.is_root() && s.output.checkpoint_interval > 0)
      std::filesystem::create_directories(s.output.checkpoint_dir);
    sub.exchange_halo(P.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(P, sub);

    if (mpi.is_root()) {
      std::printf("# nu = %.6e (Re_b = %.0f, delta = %g), dz_wall+/dz_center+ ~ %.2f/%.2f (Re_tau 180)\n",
                  nu, s.physics.reynolds_bulk, delta,
                  (faces_z[1] - faces_z[0]) * 180 / delta,
                  (faces_z[s.mesh.nz / 2 + 1] - faces_z[s.mesh.nz / 2]) * 180 / delta);
      std::printf("# step      time        dt      bulk     max|div|   u_tau   Re_tau   -dPdx    sec/step\n");
    }

    // 시간 평균 프로파일 누적 (매 스텝, stats_start_step 부터)
    std::vector<real_t> sum_u(s.mesh.nz, 0.0), sum_uu(s.mesh.nz, 0.0),
        sum_vv(s.mesh.nz, 0.0), sum_ww(s.mesh.nz, 0.0), sum_uw(s.mesh.nz, 0.0);
    long samples = 0;

    const auto write_statistics = [&](long step) {
      if (!mpi.is_root() || samples == 0) return;
      std::vector<real_t> u_mean(s.mesh.nz);
      for (int q = 0; q < s.mesh.nz; ++q) u_mean[q] = sum_u[q] / samples;
      const real_t u_tau = friction_velocity(u_mean, dmz_bottom, dmz_top, nu);
      const real_t re_tau = u_tau * delta / nu;
      char path[512];
      std::snprintf(path, sizeof(path), "%s/channel_stats_%08ld.dat",
                    s.output.stats_dir.c_str(), step);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      std::fprintf(f, "# channel statistics  step=%ld samples=%ld\n", step, samples);
      std::fprintf(f, "# Re_b=%g nu=%.8e u_tau=%.8f Re_tau=%.3f mean_dPdx=%.8e\n",
                   s.physics.reynolds_bulk, nu, u_tau, re_tau, mean_dPdx);
      std::fprintf(f, "# z  z_plus  U_mean  U_plus  u_rms  v_rms  w_rms  uw_fluct\n");
      for (int q = 0; q < s.mesh.nz; ++q) {
        const real_t z_wall = std::min(z_center[q], s.domain.lz - z_center[q]);
        const real_t mean = sum_u[q] / samples;
        const real_t uu = std::sqrt(std::max(real_t(0), sum_uu[q] / samples - mean * mean));
        const real_t vv = std::sqrt(std::max(real_t(0), sum_vv[q] / samples));
        const real_t ww = std::sqrt(std::max(real_t(0), sum_ww[q] / samples));
        // ⟨u'w'⟩ = ⟨uw⟩ − ⟨u⟩⟨w⟩ ≈ ⟨uw⟩ − 0 (⟨W⟩=0) — 총전단 선형성 검사용
        std::fprintf(f, "%12.8f %10.4f %12.8f %10.4f %12.8f %12.8f %12.8f %14.8e\n",
                     z_center[q], z_wall * u_tau / nu, mean, mean / u_tau, uu, vv, ww,
                     sum_uw[q] / samples);
      }
      std::fclose(f);
      std::printf("# stats written: %s  (u_tau=%.5f, Re_tau=%.2f)\n", path, u_tau, re_tau);
    };

    real_t t = static_cast<real_t>(restart_time);
    // 재시작 시 통계 누적은 새로 시작한다 (구간 누적 스냅샷의 빼기 병합 전제).
    std::vector<std::string> checkpoint_history;
    const auto write_restart_checkpoint = [&](long step_now, double t_now) {
      if (s.output.checkpoint_interval <= 0) return;
      char ck_path[512];
      std::snprintf(ck_path, sizeof(ck_path), "%s/channel_ck_%08ld.bin",
                    s.output.checkpoint_dir.c_str(), step_now);
      mpmstd::post::write_checkpoint(ck_path, step_now, t_now,
                                     {{"U", &U}, {"V", &V}, {"W", &W}, {"P", &P}},
                                     {{"mean_dPdx", mean_dPdx}}, sub);
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
    for (long step = restart_step + 1; step <= s.time.max_step; ++step) {
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max);

      // ── 평균 운동량 수지 진단 (budget_interval > 0 일 때만) ──
      // 정상·보존 이산화라면 한 스텝의 벌크 변화는 전부 설명돼야 한다:
      //   Δbulk(운동량 solve) = dt·[ |dPdx| − ½(τ(uⁿ) + τ(u*))/δ ]
      // 여기서 τ = u_τ² = ½ν(|U₁|/dmz)_양벽 (zero-ghost 벽의 명시 플럭스),
      // ½(uⁿ+u*) 는 델타형이 증분에 걸어주는 음해 절반까지 포함한 사다리꼴.
      // 남는 잔차 = 비발산 항(ADI 인수분해 교차항 + 사후 블록 커플링)이 먹은 양.
      // 투영은 x 주기라 0, CFR shift 는 그 자리에서 따로 찍는다.
      const bool budget = s.output.budget_interval > 0
                          && step % s.output.budget_interval == 0;
      // 이 스텝에 실제로 가해진 구동력 — correct_bulk_velocity 가 mean_dPdx 를
      // 갱신하기 전 값이어야 한다.
      const real_t force_applied = -mean_dPdx;
      real_t bulk_before = 0, tau_before = 0;
      if (budget) {
        bulk_before = mpmstd::post::bulk_velocity(U, grid, sub);
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        tau_before = friction_velocity(u_bar, dmz_bottom, dmz_top, nu);
        tau_before *= tau_before;
      }

      // ── 운동량 예측자 ──
      rhs_U.fill(0);
      rhs_V.fill(0);
      rhs_W.fill(0);
      momentum::add_advection_diffusion_u_constant_property(rhs_U, U, V, W, grid,
                                                             sub, nu, dt);
      momentum::add_advection_diffusion_v_constant_property(rhs_V, U, V, W, grid,
                                                             sub, nu, dt);
      momentum::add_advection_diffusion_w_constant_property(rhs_W, U, V, W, grid,
                                                             sub, nu, dt);
      pressure::add_gradient(rhs_U, rhs_V, rhs_W, P, grid, sub, dt);
      mpmstd::physics::add_mean_pressure_gradient(rhs_U, sub, mean_dPdx, dt);
      momentum::solve_constant_property(momentum_solver, grid, sub, rhs_U, rhs_V,
                                        rhs_W, U, V, W, nu, dt);
      sync_velocity(U, V, W, sub);

      real_t bulk_after_solve = 0, tau_after = 0;
      if (budget) {
        bulk_after_solve = mpmstd::post::bulk_velocity(U, grid, sub);
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        tau_after = friction_velocity(u_bar, dmz_bottom, dmz_top, nu);
        tau_after *= tau_after;
      }

      // ── CFR: 벌크 속도 유지 (ghost 최신 상태에서 측정 — channel_forcing.hpp ⚠) ──
      mpmstd::physics::correct_bulk_velocity(U, grid, sub, bulk_target, dt, mean_dPdx);
      sync_velocity(U, V, W, sub);
      const real_t bulk_after_cfr =
          budget ? mpmstd::post::bulk_velocity(U, grid, sub) : real_t(0);

      // ── 증분 투영 ──
      rhs_P.fill(0);
      pressure::add_divergence(rhs_P, U, V, W, grid, sub, dt);
      pressure::solve(pressure_solver, grid, sub, rhs_P, dP);
      pressure::correct_velocity(U, V, W, dP, grid, sub, dt);
      pressure::accumulate(P, dP);
      sync_velocity(U, V, W, sub);  // P 는 accumulate 가 ghost 포함이라 재교환 불요

      t += dt;
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_restart_checkpoint(step, t);

      if (budget) {
        const real_t bulk_after_projection = mpmstd::post::bulk_velocity(U, grid, sub);
        const real_t drag = real_t(0.5) * (tau_before + tau_after) / delta;
        const real_t expected = dt * (force_applied - drag);
        const real_t measured = bulk_after_solve - bulk_before;
        if (mpi.is_root())
          std::printf("# budget %7ld  force=%+.6e drag=%+.6e expected=%+.6e "
                      "solve=%+.6e leak=%+.6e (%+.2f%% of force)  cfr=%+.6e "
                      "projection=%+.6e\n",
                      step, force_applied, drag, expected, measured,
                      (measured - expected) / dt,
                      100 * (measured - expected) / (dt * force_applied),
                      bulk_after_cfr - bulk_after_solve,
                      bulk_after_projection - bulk_after_cfr);
      }

      // ── 통계 (매 스텝) ──
      if (step >= s.output.stats_start_step) {
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        const auto uu = mpmstd::post::plane_average_square(U, sub);
        const auto vv = mpmstd::post::plane_average_square(V, sub);
        const auto ww = mpmstd::post::plane_average_square(W, sub);
        const auto uw = mpmstd::post::plane_average_product(U, W, sub);
        for (int q = 0; q < s.mesh.nz; ++q) {
          sum_u[q] += u_bar[q];
          sum_uu[q] += uu[q];
          sum_vv[q] += vv[q] - 0;  // ⟨V⟩≈0 — rms 는 ⟨V²⟩ 로 충분
          sum_ww[q] += ww[q];
          sum_uw[q] += uw[q];      // 레이놀즈 전단 ⟨uw⟩ (⟨W⟩≈0 — 평균 몫 생략)
        }
        ++samples;
        if ((step - s.output.stats_start_step) % s.output.stats_write_interval == 0
            && step > s.output.stats_start_step)
          write_statistics(step);
      }

      // ── 모니터 ──
      if (step % s.output.monitor_interval == 0 || step == 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        const real_t bulk = mpmstd::post::bulk_velocity(U, grid, sub);
        const auto u_bar = mpmstd::post::plane_average(U, sub);
        const real_t u_tau = friction_velocity(u_bar, dmz_bottom, dmz_top, nu);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %9.6f %10.2e %8.5f %7.2f %9.6f %8.3f\n",
                      step, t, dt, bulk, div, u_tau, u_tau * delta / nu, -mean_dPdx,
                      (wall_now - wall_prev) / s.output.monitor_interval);
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        if (!(div == div) || div > 1e3)
          throw std::runtime_error("divergence blew up — aborting");
      }
    }

    write_statistics(s.time.max_step);
    if (s.output.checkpoint_interval > 0
        && s.time.max_step % s.output.checkpoint_interval != 0)
      write_restart_checkpoint(s.time.max_step, t);  // 연장 런용 마지막 상태
    if (mpi.is_root() && samples > 0) {
      std::vector<real_t> u_mean(s.mesh.nz);
      for (int q = 0; q < s.mesh.nz; ++q) u_mean[q] = sum_u[q] / samples;
      const real_t u_tau = friction_velocity(u_mean, dmz_bottom, dmz_top, nu);
      const real_t re_tau = u_tau * delta / nu;
      // 목표 Re_tau 는 케이스(덱)마다 다르다 — 판정은 plot 노트북이 참조
      // 파일의 실측 Re_tau 와 비교한다 (180 덱=LM 182.1, 550 덱=LM 543.5).
      std::printf("# FINAL: Re_tau = %.2f  (Re_b = %g)\n", re_tau,
                  s.physics.reynolds_bulk);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    return 1;
  }
  return 0;
}
