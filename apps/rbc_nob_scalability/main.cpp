// rbc_nob 확장성 측정 드라이버 — rbc_nob 시간 루프의 main 호출 단위 구간 시간 (docs/Scalability/plan.md §1·§2).
//
// 시간 루프는 apps/rbc_nob/main.cpp 와 **같은 호출을 같은 순서로** 부른다 (수치 불변 — 구간 경계에서
// 동기화만 더한다). 빠진 것: 재시작·체크포인트·통계 누적(plan.md §1 의 "제외"). 그래서 모니터 줄은
// stats_start_time 을 끈 rbc_nob 런과 같아야 하고, job_scal.sh 의 게이트가 매 잡마다 그것을 확인한다.
// rbc_nob 루프를 고치면 여기도 같이 고친다.
//
//   사용: rbc_nob_scalability <input.ini> [timing.csv]
//   [timing] warmup_steps 스텝 뒤부터 누적 → 런 끝에 구간 표(stdout) + CSV(두 번째 인자를 줄 때).
//   구간 이름 = plan.md §1 표 (dt_cfl … monitor) + step_total, 분산 TDMA 의 MPI 시간 tdma_comm_<풀이>_<축>.
#include <mpi.h>
#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "boundary/wall.hpp"
#include "common/phase_timer.hpp"
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
#include "physics/properties.hpp"
#include "post/statistics.hpp"
#include "rbc_nob/statistics.hpp"
#include "rbc_nob_scalability/settings.hpp"
#include "time/time_step.hpp"
#include "time/viscous_time_step.hpp"

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

constexpr real_t kWallTemperatureBottom = 0.5;  // rbc_nob 과 같은 벽 온도
constexpr real_t kWallTemperatureTop = -0.5;

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

// IC — rbc_nob 냉시작과 같다: 전도 선형 T(z) + 다중모드 W 섭동.
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

// MPMSTD_MEM_REPORT=1 이면 초기화 단계마다 GPU 메모리 사용량(cudaMemGetInfo 의 total − free, 랭크 max)을 찍는다.
// 직전 표시 이후 증가분도 같이 — 어느 할당이 큰지 보는 진단용. 미설정이면 아무것도 안 한다.
void mem_mark(const char* label, MPI_Comm comm, const bool root) {
#ifdef MPMSTD_GPU
  static const bool enabled = [] {
    const char* v = std::getenv("MPMSTD_MEM_REPORT");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  static double previous = 0;
  if (!enabled) return;
  cudaDeviceSynchronize();
  std::size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  double used = static_cast<double>(total_bytes - free_bytes) / 1e9, max_used = 0;
  MPI_Allreduce(&used, &max_used, 1, MPI_DOUBLE, MPI_MAX, comm);
  if (root) std::printf("# mem %-26s %8.2f GB  (+%7.2f)\n", label, max_used, max_used - previous);
  previous = max_used;
#else
  (void)label; (void)comm; (void)root;
#endif
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
  if (argc < 2 || argc > 3) {
    if (mpi.is_root()) std::fprintf(stderr, "usage: %s <input.ini> [timing.csv]\n", argv[0]);
    return 1;
  }
  try {
    const rbc_nob_scalability::Settings s = rbc_nob_scalability::load_settings(argv[1]);
    const std::string csv_path = argc == 3 ? argv[2] : "";
    const bool constant_property_limit = s.physics.delta_temperature == 0;
    if (mpi.is_root()) {
      std::printf("# settings loaded from %s  (%d ranks, tdma=%s)\n", argv[1], mpi.world_size(),
                  mpmstd::linear_solver::tdma_backend_name());
      std::fputs(rbc_nob_scalability::format_settings(s).c_str(), stdout);
    }

    const Topology topo(mpi, s.process.npx, s.process.npy, s.process.npz,
                        /*periodic_x=*/true, /*periodic_y=*/true, /*periodic_z=*/false);
    const Subdomain sub(topo, s.mesh.nx, s.mesh.ny, s.mesh.nz);
    mem_mark("context + topology", mpi.world_comm(), mpi.is_root());
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

    const real_t Cmu = std::sqrt(s.physics.prandtl / s.physics.rayleigh);
    const real_t Ct = 1.0 / std::sqrt(s.physics.rayleigh * s.physics.prandtl);
    const physics::PropertyModel model =
        constant_property_limit ? physics::PropertyModel{}
                                : physics::PropertyModel::glycerol(s.physics.delta_temperature);
    const real_t density_quadratic_ratio = model.density_quadratic_ratio();

    // 필드·솔버 — rbc_nob 과 같은 할당 (rhs_U 를 압력 우변·dP 로 재사용하는 것까지 같다)
    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"), T(nx, ny, nz, "T"), P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"), rhs_W(nx, ny, nz, "rhs_W");
    Field& rhs_dP = rhs_U;
    Field& dP = rhs_dP;
    Field pressure_predictor(nx, ny, nz, "dPhat"), previous_predictor(nx, ny, nz, "dP0");
    mem_mark("app fields (10)", mpi.world_comm(), mpi.is_root());
    physics::PropertyFields properties(nx, ny, nz);
    mem_mark("property fields (6)", mpi.world_comm(), mpi.is_root());
    mpmstd::equation::Workspace workspace(std::max({momentum::Solver::required_slot_elements(sub),
                                                    scalar::Solver::required_slot_elements(sub),
                                                    pressure::Solver::required_slot_elements(grid, sub)}));
    mem_mark("workspace (5 slots)", mpi.world_comm(), mpi.is_root());
    momentum::Solver momentum_solver(sub, workspace);
    mem_mark("momentum solver", mpi.world_comm(), mpi.is_root());
    scalar::Solver scalar_solver(sub, workspace);
    mem_mark("scalar solver", mpi.world_comm(), mpi.is_root());
    pressure::Solver pressure_solver(grid, sub, workspace);
    mem_mark("pressure solver", mpi.world_comm(), mpi.is_root());

    conduction_initial_condition(W, T, grid, sub, s.domain.lx, s.domain.ly, s.domain.lz,
                                 s.initial.perturb);
    sync_velocity(U, V, W, sub);
    sync_temperature(T, sub);
    sub.exchange_halo(P.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(P, sub);
    sub.exchange_halo(pressure_predictor.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(pressure_predictor, sub);
    sub.exchange_halo(previous_predictor.data());
    mpmstd::boundary::fill_z_wall_zero_gradient(previous_predictor, sub);

    // 모니터 진단용 (θ_c·벽 Nu) — 누적은 하지 않으므로 Nu 열(체적 Nu)은 rbc_nob 의 통계 끈 런처럼 0
    rbc_nob::StatisticsConfig stats_config;
    stats_config.nz = s.mesh.nz;
    stats_config.lx = s.domain.lx;
    stats_config.ly = s.domain.ly;
    stats_config.lz = s.domain.lz;
    stats_config.rayleigh = s.physics.rayleigh;
    stats_config.prandtl = s.physics.prandtl;
    stats_config.delta_temperature = s.physics.delta_temperature;
    stats_config.Cmu = Cmu;
    stats_config.Ct = Ct;
    stats_config.model = model;
    stats_config.z_center = z_center;
    stats_config.dz = dz_global;
    stats_config.wall_temperature_bottom = kWallTemperatureBottom;
    stats_config.wall_temperature_top = kWallTemperatureTop;
    rbc_nob::Statistics statistics(stats_config, sub);
    mem_mark("statistics", mpi.world_comm(), mpi.is_root());

    if (mpi.is_root()) {
      std::printf("# %s: Ra=%.3g Pr=%.3g DeltaT=%.1fK | %dx%dx%d on %dx%dx%d ranks, "
                  "warmup %ld of %ld steps\n",
                  constant_property_limit ? "constant-property limit" : "NOB glycerol",
                  s.physics.rayleigh, s.physics.prandtl, s.physics.delta_temperature,
                  s.mesh.nx, s.mesh.ny, s.mesh.nz, s.process.npx, s.process.npy, s.process.npz,
                  s.timing.warmup_steps, s.time.max_step);
      std::printf("# step      time        dt     max|div|   theta_c      Nu      "
                  "Tmin     Tmax    Nu_b     Nu_t     D_x    u_bot_max  u_top_max  sec/step\n");
    }
    physics::update_properties(properties, T, model, sub);

    // ── 구간 타이머. 워밍업 동안은 비활성(begin/end no-op) ──
    mpmstd::common::PhaseTimer timer;
    if (s.timing.solver_phases) {  // 풀이 안의 호출별 구간 (plan.md §1.1)
      momentum_solver.timer = &timer;
      scalar_solver.timer = &timer;
      pressure_solver.timer = &timer;
    }
    // 분산 TDMA 의 MPI(Alltoallv) 누적 초 — 플랜의 comm_seconds 훅(측정 오버헤드 없음). 워밍업 끝에 영점을
    // 잡고 런 끝에 구간처럼 싣는다. 분할하지 않은 축은 0.
    struct CommProbe {
      const char* name;
      const double* seconds;
      double base;
    };
    std::vector<CommProbe> comm_probes = {
        {"tdma_comm_scalar_x", &scalar_solver.plan_x.comm_seconds, 0},
        {"tdma_comm_scalar_y", &scalar_solver.plan_y.comm_seconds, 0},
        {"tdma_comm_scalar_z", &scalar_solver.plan_z.comm_seconds, 0},
        {"tdma_comm_momentum_x", &momentum_solver.plan_x.comm_seconds, 0},
        {"tdma_comm_momentum_y", &momentum_solver.plan_y.comm_seconds, 0},
        {"tdma_comm_momentum_z", &momentum_solver.plan_z.comm_seconds, 0},
        {"tdma_comm_pressure_z", &pressure_solver.plan_z.comm_seconds, 0},
    };
    // 압력 펜슬 전치 4 종의 MPI(Alltoallv)만 — 이름에 '.' 을 넣어 top-level 구간 합에서 빠지게 (pressure_solve 안의 몫)
    if (pressure_solver.pencil) {
      auto& L = *pressure_solver.pencil;
      comm_probes.push_back({"pressure.comm.block_to_lines", &L.block_to_lines.comm_seconds, 0});
      comm_probes.push_back({"pressure.comm.x_to_y", &L.x_to_y_spectrum.comm_seconds, 0});
      comm_probes.push_back({"pressure.comm.y_to_x", &L.y_to_x_spectrum.comm_seconds, 0});
      comm_probes.push_back({"pressure.comm.lines_to_block", &L.lines_to_block.comm_seconds, 0});
    }
    if (pressure_solver.planes) {   // 평면 묶음 경로 (전치 2 회)
      auto& L = *pressure_solver.planes;
      comm_probes.push_back({"pressure.comm.block_to_planes", &L.block_to_planes.comm_seconds, 0});
      comm_probes.push_back({"pressure.comm.planes_to_block", &L.planes_to_block.comm_seconds, 0});
    }

    real_t t = 0;
    long step = 0;
    rhs_dP.fill(0);   // ghost 는 여기서 한 번만 0 (rbc_nob 과 같다)
    double wall_prev = MPI_Wtime();
    while (step < s.time.max_step) {
      if (!timer.active() && step >= s.timing.warmup_steps) {
        timer.activate();
        for (auto& probe : comm_probes) probe.base = *probe.seconds;
      }
      timer.begin("step_total");
      timer.begin("dt_cfl");
      real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max);
      timer.end("dt_cfl");
      timer.begin("dt_viscous");
      if (s.time.viscous_coupling_number > 0)
        dt = std::min(dt, mpmstd::time_integration::compute_viscous_coupling_dt(
                              properties.viscosity, grid, sub, Cmu,
                              static_cast<real_t>(s.time.viscous_coupling_number)).dt);
      timer.end("dt_viscous");

      timer.begin("properties_1");
      physics::update_properties(properties, T, model, sub);
      timer.end("properties_1");

      timer.begin("scalar_solve");
      scalar::solve_variable_property(scalar_solver, grid, sub, T, U, V, W, properties, Ct, dt);
      timer.end("scalar_solve");
      timer.begin("sync_T");
      sync_temperature(T, sub);
      timer.end("sync_T");

      timer.begin("momentum_rhs");
      momentum::build_variable_property_rhs(rhs_U, rhs_V, rhs_W, U, V, W, T, P,
                                            properties.viscosity, properties.inverse_density,
                                            grid, sub, Cmu, density_quadratic_ratio,
                                            s.physics.delta_temperature, dt);
      timer.end("momentum_rhs");
      timer.begin("momentum_solve");
      momentum::solve_variable_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V,
                                        W, properties.viscosity, properties.inverse_density,
                                        Cmu, dt);
      timer.end("momentum_solve");
      timer.begin("sync_UVW_1");
      sync_velocity(U, V, W, sub);
      timer.end("sync_UVW_1");

      timer.begin("properties_2");
      physics::update_properties(properties, T, model, sub);
      timer.end("properties_2");
      timer.begin("pressure_rhs");
      pressure::build_variable_property_rhs(rhs_dP, U, V, W, pressure_predictor,
                                            properties.inverse_density, grid, sub, dt);
      timer.end("pressure_rhs");
      timer.begin("pressure_solve");
      pressure::solve(pressure_solver, grid, sub, rhs_dP, dP);
      timer.end("pressure_solve");
      timer.begin("pressure_correct");
      pressure::correct_velocity_and_accumulate_variable_property(
          U, V, W, P, dP, pressure_predictor, properties.inverse_density, grid, sub, dt);
      timer.end("pressure_correct");
      timer.begin("sync_UVW_2");
      sync_velocity(U, V, W, sub);
      timer.end("sync_UVW_2");

      timer.begin("predictor");
      pressure::advance_pressure_predictor(pressure_predictor, previous_predictor, dP);
      sub.exchange_halo(pressure_predictor.data());
      mpmstd::boundary::fill_z_wall_zero_gradient(pressure_predictor, sub);
      timer.end("predictor");

      timer.begin("clip");
      mpmstd::equation::clip_interior(T, sub, kWallTemperatureTop,
                                      kWallTemperatureBottom);
      timer.end("clip");

      t += dt;
      ++step;

      if (step % s.output.monitor_interval == 0 || step == 1) {
        timer.begin("monitor");
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
        const auto viscous = mpmstd::time_integration::compute_viscous_coupling_dt(
            properties.viscosity, grid, sub, Cmu, real_t(1), dt);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %9.5f %8.4f %8.4f %8.4f %8.4f %8.4f %7.2f %10.3e %10.3e %8.3f\n",
                      step, t, dt, div, theta_c_now, statistics.nusselt_volume(), t_min, t_max,
                      nu_bottom_now, nu_top_now, viscous.mixed_number, u_abs_max[0],
                      u_abs_max[s.mesh.nz - 1],
                      (wall_now - wall_prev) / s.output.monitor_interval);
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        timer.end("monitor");
        if (!(div == div) || div > 1e3)
          throw std::runtime_error("divergence blew up — aborting");
      }
      timer.end("step_total");
      timer.count_step();
      if (step == 1) mem_mark("after step 1", mpi.world_comm(), mpi.is_root());
    }

    for (const auto& probe : comm_probes) timer.set_external(probe.name, *probe.seconds - probe.base);
    const std::string decomposition = std::to_string(s.process.npx) + "x" +
                                      std::to_string(s.process.npy) + "x" +
                                      std::to_string(s.process.npz);
    timer.report(sub.topology().cart_comm(), csv_path,
                 {{"nx", std::to_string(s.mesh.nx)}, {"ny", std::to_string(s.mesh.ny)},
                  {"nz", std::to_string(s.mesh.nz)}, {"decomposition", decomposition},
                  {"ranks", std::to_string(mpi.world_size())},
                  {"tdma", mpmstd::linear_solver::tdma_backend_name()},
                  {"warmup_steps", std::to_string(s.timing.warmup_steps)},
                  {"max_step", std::to_string(s.time.max_step)},
                  {"solver_phases", s.timing.solver_phases ? "1" : "0"}});
    if (mpi.is_root() && !csv_path.empty()) std::printf("# timing csv: %s\n", csv_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    return 1;
  }
  return 0;
}
