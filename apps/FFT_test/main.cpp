// FFT_test — FFTU(순환 분배 단일 all-to-all FFT, Koopman & Bisseling 2023) 기반 압력 Poisson 솔버를
// 기존 equation::pressure::Solver 와 같은 문제·같은 프로세스 격자에서 비교한다 (apps/FFT_test/plan.md).
//
//   사용: FFT_test <input.ini> [csv_prefix]
//     1) 순환 FFT 단위 시험       ([check] fft)       실수 평면파 → 스펙트럼 두 점, 정·역 왕복
//     2) Poisson 정확도           ([check] accuracy)  제조해 이산 일관성, 난수 rhs 잔차, 두 해 대조
//     3) 속도 (bench.solves > 0)   같은 난수 rhs 를 기존 → 새 순서로 교대해 풀며 ms/solve (랭크 max)
//        - 총시간 패스(타이머 경계 = 풀이 하나) → <csv_prefix>_total.csv
//        - 단계 패스([bench] phases)  → <csv_prefix>_phases.csv  (pressure.* = 기존, fftu.* = 새)
//     정확도 결과는 <csv_prefix>_accuracy.csv 와 로그 끝 PASS/FAIL.
//     [check] contract = true 면 3) 대신 CaNS·PaScaL 비교 모드 (plan.md §10): 같은 해석 문제를 MPM-STD 평면묶음·펜슬,
//     FFTU 로 풀어 해석해 RMS 를 대조하고, 같은 시간 정의(동기화 → 풀이 → 동기화, 풀이별 rank-max 의 평균)로 잰다.
//       <csv_prefix>_times.csv, <csv_prefix>_<방법>_phases.csv
//
// 기존 소스는 include 만 한다 (수정 없음). 기존 솔버는 npx·npy 로 경로를 스스로 고른다 (slab / plane / pencil).

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

#include "../framework.hpp"
#include "checks.hpp"
#include "common/phase_timer.hpp"
#include "cyclic_poisson.hpp"
#include "device_loop.hpp"
#include "equation/pressure/solve.hpp"
#include "equation/workspace.hpp"
#include "exchange.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"
#include "settings.hpp"
#include "transpose_transport.hpp"

using mpmstd::real_t;
using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;
using mpmstd::parallel::Topology;

namespace {

const char* old_path_name(const mpmstd::equation::pressure::Solver& solver) {
  if (solver.planes) return "plane";
  if (solver.pencil) return "pencil";
  return "slab";
}

// 기존 솔버의 x·y 전치가 MPI_Alltoallv 안에서 보낸 누적 초 (슬랩 경로는 전치가 없어 0)
double old_transpose_comm_seconds(const mpmstd::equation::pressure::Solver& solver) {
  double seconds = 0;
  if (solver.planes)
    seconds += solver.planes->block_to_planes.comm_seconds + solver.planes->planes_to_block.comm_seconds;
  if (solver.pencil)
    seconds += solver.pencil->block_to_lines.comm_seconds + solver.pencil->lines_to_block.comm_seconds
               + solver.pencil->x_to_y_spectrum.comm_seconds + solver.pencil->y_to_x_spectrum.comm_seconds;
  return seconds;
}

// CaNS·PaScaL 비교 모드의 한 방법 — 같은 rhs 를 푸는 풀이 하나와 그 측정 훅.
struct Method {
  std::string name;
  std::function<void()> solve;
  std::function<void(mpmstd::common::PhaseTimer*)> attach_timer;
  std::function<double()> tdma_comm_seconds;
  std::function<double()> transpose_comm_seconds;   // FFTU 는 fftu.*_mpi·a2a 구간으로 재므로 0
};

struct TimedBlock {
  double mean_rank_max = 0, mean_rank_average = 0, best_rank_max = 0, worst_rank_max = 0;   // ms/solve
};

// CaNS·PaScaL 벤치와 같은 정의: 동기화·배리어 뒤 풀이마다 시작 → 풀이 → 동기화 → 끝, 풀이별 rank-max 의 평균.
TimedBlock timed_block(Method& method, const int solves, MPI_Comm comm) {
  std::vector<double> seconds(solves), rank_max(solves), rank_sum(solves);
  fft_test::synchronize_device();
  MPI_Barrier(comm);
  for (int n = 0; n < solves; ++n) {
    const double t0 = MPI_Wtime();
    method.solve();
    fft_test::synchronize_device();
    seconds[n] = MPI_Wtime() - t0;
  }
  MPI_Allreduce(seconds.data(), rank_max.data(), solves, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(seconds.data(), rank_sum.data(), solves, MPI_DOUBLE, MPI_SUM, comm);
  int ranks = 1;
  MPI_Comm_size(comm, &ranks);
  TimedBlock out;
  out.best_rank_max = 1e3 * *std::min_element(rank_max.begin(), rank_max.end());
  out.worst_rank_max = 1e3 * *std::max_element(rank_max.begin(), rank_max.end());
  for (int n = 0; n < solves; ++n) {
    out.mean_rank_max += 1e3 * rank_max[n] / solves;
    out.mean_rank_average += 1e3 * rank_sum[n] / (ranks * solves);
  }
  return out;
}

// 통신자 안에서 나와 같은 노드에 있는 랭크 수 — 실제 랭크 배치를 로그에 남긴다.
int ranks_on_my_node(MPI_Comm comm) {
  MPI_Comm node = MPI_COMM_NULL;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int size = 1;
  MPI_Comm_size(node, &size);
  MPI_Comm_free(&node);
  return size;
}

void print_memory(const char* label, const bool root) {
#ifdef MPMSTD_GPU
  std::size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  if (root)
    std::printf("# memory %-28s used %.2f GB of %.2f GB (rank 0)\n", label,
                (total_bytes - free_bytes) / 1e9, total_bytes / 1e9);
#else
  (void)label;
  (void)root;
#endif
}

void write_accuracy_csv(const std::string& path, const fft_test::FftCheck& f, const fft_test::PoissonCheck& p,
                        const std::vector<std::pair<std::string, std::string>>& header) {
  FILE* csv = std::fopen(path.c_str(), "w");
  if (!csv) return;
  for (const auto& kv : header) std::fprintf(csv, "# %s=%s\n", kv.first.c_str(), kv.second.c_str());
  std::fprintf(csv, "quantity,value\n");
  std::fprintf(csv, "fft_delta_error,%.6e\nfft_roundtrip_error,%.6e\n", f.delta_error, f.roundtrip_error);
  std::fprintf(csv, "manufactured_new,%.6e\nmanufactured_old,%.6e\n", p.manufactured_new, p.manufactured_old);
  std::fprintf(csv, "residual_new,%.6e\nresidual_old,%.6e\ndifference,%.6e\n", p.residual_new, p.residual_old,
               p.difference);
  std::fclose(csv);
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
  const bool root = mpi.is_root();
  std::setvbuf(stdout, nullptr, _IOLBF, 0);   // 파일로 보내도 줄마다 내보낸다 — 멈춘 런의 위치가 로그에 남게
  // 진행 표시: 잡이 멈추면 로그의 마지막 stage 가 멈춘 곳이다
  const double t_start = MPI_Wtime();
  const auto stage = [&](const char* what) {
    if (root) std::printf("# stage %8.2f s  %s\n", MPI_Wtime() - t_start, what);
  };
  if (argc < 2 || argc > 3) {
    if (root) std::fprintf(stderr, "usage: %s <input.ini> [csv_prefix]\n", argv[0]);
    return 1;
  }
  try {
    const fft_test::Settings s = fft_test::load_settings(argv[1]);
    const std::string prefix = argc == 3 ? argv[2] : "";
    if (root) {
      std::printf("# FFT_test  settings from %s  (%d ranks, tdma=%s)\n", argv[1], mpi.world_size(),
                  mpmstd::linear_solver::tdma_backend_name());
      std::fputs(fft_test::format_settings(s).c_str(), stdout);
    }

    const Topology topo(mpi, s.process.npx, s.process.npy, s.process.npz, /*periodic_x=*/true,
                        /*periodic_y=*/true, /*periodic_z=*/false);
    const Subdomain sub(topo, s.mesh.nx, s.mesh.ny, s.mesh.nz);
    const auto faces_z = s.stretch.uniform_z
                             ? mpmstd::grid::make_uniform_faces(s.mesh.nz, s.domain.lz)
                             : mpmstd::grid::make_tanh_faces(s.mesh.nz, s.domain.lz, s.stretch.gamma_z);
    const mpmstd::grid::Grid grid{
        mpmstd::grid::make_axis(mpmstd::grid::make_uniform_faces(s.mesh.nx, s.domain.lx),
                                {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
        mpmstd::grid::make_axis(mpmstd::grid::make_uniform_faces(s.mesh.ny, s.domain.ly),
                                {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
        mpmstd::grid::make_axis(faces_z, {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
    };
    const int tx = sub.x().n_total, ty = sub.y().n_total, tz = sub.z().n_total;
    Field rhs(tx, ty, tz, "rhs"), phi_new(tx, ty, tz, "phi_new"), phi_old(tx, ty, tz, "phi_old");
    print_memory("fields", root);

    stage("create FFTU solver");
    fft_test::CyclicPoissonSolver new_solver(grid, sub);
    print_memory("+ cyclic solver", root);
    stage("create existing solver");
    // [bench] only: 쉼표로 여러 방법을 고른다. 비우면 전부
    const auto wanted = [&s](const std::string& name) {
      return s.bench.only.empty() || ("," + s.bench.only + ",").find("," + name + ",") != std::string::npos;
    };
    // 비교 모드에서는 펜슬 경로도 만든다. 경로는 생성 때 환경변수로 정해지고, 풀이 때는 다시 읽지 않는다.
    // 두 경로는 차례로 도니 Workspace 하나를 나눠 쓴다.
    const bool want_pencil = s.check.contract && s.process.npx * s.process.npy > 1;
    std::size_t slot_elements = mpmstd::equation::pressure::Solver::required_slot_elements(grid, sub);
    if (want_pencil) {
      setenv("MPMSTD_PRESSURE_PENCIL", "transpose4", 1);
      slot_elements = std::max(slot_elements, mpmstd::equation::pressure::Solver::required_slot_elements(grid, sub));
      unsetenv("MPMSTD_PRESSURE_PENCIL");
    }
    mpmstd::equation::Workspace workspace(slot_elements);
    mpmstd::equation::pressure::Solver old_solver(grid, sub, workspace);
    std::unique_ptr<mpmstd::equation::pressure::Solver> pencil_solver;
    if (want_pencil) {
      setenv("MPMSTD_PRESSURE_PENCIL", "transpose4", 1);
      pencil_solver = std::make_unique<mpmstd::equation::pressure::Solver>(grid, sub, workspace);
      unsetenv("MPMSTD_PRESSURE_PENCIL");
      // 통신 방식 시험 — 등록한 통신자만 바뀐다 (transpose_transport.hpp). 재는 경로만 등록한다.
      if (pencil_solver->pencil && wanted("mpm_pencil")) {
        fft_test::register_transpose_comm(pencil_solver->pencil->comm);
        fft_test::register_tdma_comm(pencil_solver->plan_z.backend.comm);
      }
      if (old_solver.planes && wanted("mpm_plane")) {
        fft_test::register_transpose_comm(old_solver.planes->comm);
        fft_test::register_tdma_comm(old_solver.plan_z.backend.comm);
      }
    }
    print_memory("+ existing solver", root);

    const fft_test::CyclicDims& d = new_solver.layout().dims();
    if (root) {
      std::printf("# grid %dx%dx%d  process %dx%dx%d  z %s (gamma %.3g)\n", s.mesh.nx, s.mesh.ny, s.mesh.nz,
                  s.process.npx, s.process.npy, s.process.npz, s.stretch.uniform_z ? "uniform" : "tanh",
                  s.stretch.gamma_z);
      std::printf("# cyclic: q %d  local y m %d  packet c %d  x modes %d  nz %d  (z chain %d)\n", d.q, d.m, d.c,
                  d.n_mode_x, d.nz, topo.z().nprocs);
      std::printf("# existing solver path: %s\n", old_path_name(old_solver));
      std::printf("# MPI mode: %s, FFTU peer exchange: %s, x-y transpose: %s, z TDMA all-to-all: %s\n",
                  new_solver.fft().host_staging() ? "host staging" : "device buffers", fft_test::a2a_mode_name(),
                  fft_test::transpose_mode_name(), fft_test::tdma_mode_name());
    }
    const std::vector<std::pair<std::string, std::string>> header = {
        {"nx", std::to_string(s.mesh.nx)},       {"ny", std::to_string(s.mesh.ny)},
        {"nz", std::to_string(s.mesh.nz)},       {"npx", std::to_string(s.process.npx)},
        {"npy", std::to_string(s.process.npy)},  {"npz", std::to_string(s.process.npz)},
        {"np", std::to_string(mpi.world_size())}, {"q", std::to_string(d.q)},
        {"uniform_z", s.stretch.uniform_z ? "1" : "0"},
        {"gamma_z", std::to_string(s.stretch.gamma_z)}, {"old_path", old_path_name(old_solver)},
        {"mpi_mode", new_solver.fft().host_staging() ? "host" : "device"},
        {"a2a_mode", fft_test::a2a_mode_name()},
        {"transpose_mode", fft_test::transpose_mode_name()},
        {"tdma_a2a_mode", fft_test::tdma_mode_name()}};

    const real_t tol = s.check.tolerance;
    // GTL 첫 디바이스 집합통신 예열 (노드 간 첫 호출이 틀린 값을 낸 전례) — 결과는 버린다
    stage("warm-up: fill rhs");
    fft_test::fill_random_rhs(rhs, grid, sub);
    stage("warm-up: first FFTU solve");
    new_solver.solve(rhs, phi_new);
    stage("warm-up: first existing solve");
    mpmstd::equation::pressure::solve(old_solver, grid, sub, rhs, phi_old);

    fft_test::FftCheck fft_check;
    if (s.check.fft) {
      stage("fft check");
      fft_check = fft_test::check_cyclic_fft(new_solver.fft(), topo.cart_comm());
      if (root)
        std::printf("fft check (%d plane waves): delta %.3e  roundtrip %.3e\n", fft_check.modes_tested,
                    fft_check.delta_error, fft_check.roundtrip_error);
      CHECK(fft_check.delta_error < tol);
      CHECK(fft_check.roundtrip_error < tol);
    }
    fft_test::PoissonCheck poisson_check;
    if (s.check.accuracy) {
      stage("accuracy check");
      poisson_check = fft_test::check_poisson(new_solver, old_solver, grid, sub, s.domain.lx, s.domain.ly,
                                              s.domain.lz, rhs, phi_new, phi_old);
      if (root) {
        std::printf("manufactured (discrete rhs): new %.3e  old %.3e   (max|phi_h - phi| / max|phi|)\n",
                    poisson_check.manufactured_new, poisson_check.manufactured_old);
        std::printf("random rhs residual:         new %.3e  old %.3e   (max|L phi - rhs| / max|rhs|)\n",
                    poisson_check.residual_new, poisson_check.residual_old);
        std::printf("new vs old:                  %.3e               (max|phi_new - phi_old| / max|phi_old|)\n",
                    poisson_check.difference);
      }
      CHECK(poisson_check.manufactured_new < tol);
      CHECK(poisson_check.residual_new < tol);
      CHECK(poisson_check.difference < tol);
    }
    if (root && !prefix.empty() && (s.check.fft || s.check.accuracy))
      write_accuracy_csv(prefix + "_accuracy.csv", fft_check, poisson_check, header);

    if (s.check.contract) {
      std::vector<Method> methods;
      const auto add_existing = [&](mpmstd::equation::pressure::Solver& solver, const std::string& name) {
        methods.push_back({name, [&grid, &sub, &rhs, &phi_old, &solver] { mpmstd::equation::pressure::solve(solver, grid, sub, rhs, phi_old); },
                           [&solver](mpmstd::common::PhaseTimer* t) { solver.timer = t; },
                           [&solver] { return solver.plan_z.comm_seconds; },
                           [&solver] { return old_transpose_comm_seconds(solver); }});
      };
      add_existing(old_solver, std::string("mpm_") + old_path_name(old_solver));
      if (pencil_solver && old_solver.planes) add_existing(*pencil_solver, "mpm_pencil");
      methods.push_back({"fftu", [&rhs, &phi_new, &new_solver] { new_solver.solve(rhs, phi_new); },
                         [&new_solver](mpmstd::common::PhaseTimer* t) { new_solver.timer = t; },
                         [&new_solver] { return new_solver.tdma_comm_seconds(); }, [] { return 0.0; }});

      if (!s.bench.only.empty()) {
        methods.erase(std::remove_if(methods.begin(), methods.end(), [&](const Method& m) { return !wanted(m.name); }),
                      methods.end());
        if (methods.empty()) throw std::invalid_argument("[bench] only: no method named " + s.bench.only);
      }

      // 실제 랭크 배치 (같은 노드의 랭크 수, 전 랭크 최소·최대)
      int placement[2] = {ranks_on_my_node(new_solver.layout().group()), ranks_on_my_node(topo.z().comm)}, low[2], high[2];
      MPI_Allreduce(placement, low, 2, MPI_INT, MPI_MIN, topo.cart_comm());
      MPI_Allreduce(placement, high, 2, MPI_INT, MPI_MAX, topo.cart_comm());
      if (root)
        std::printf("# placement: xy group %d ranks, %d..%d on one node; z chain %d ranks, %d..%d on one node\n",
                    new_solver.layout().group_size(), low[0], high[0], topo.z().nprocs, low[1], high[1]);

      stage("contract: fill rhs");
      fft_test::fill_contract_rhs(rhs, grid, sub);
      // 스트레칭 z 에는 해석 RMS 공식이 없다: 크게 틀린 해만 여기서 잡고, 방식 간 RMS 대조는 plot_cube.py 가 한다
      const double expected_rms = s.stretch.uniform_z ? fft_test::contract_expected_rms(grid) : std::nan("");
      stage("contract: warm-up");
      for (Method& method : methods)
        for (int w = 0; w < s.bench.warmup; ++w) method.solve();
      FILE* csv = (root && !prefix.empty()) ? std::fopen((prefix + "_times.csv").c_str(), "w") : nullptr;
      if (csv) {
        for (const auto& kv : header) std::fprintf(csv, "# %s=%s\n", kv.first.c_str(), kv.second.c_str());
        std::fprintf(csv, "# expected_rms=%.10e\n", expected_rms);
        std::fprintf(csv, "method,mean_rank_max_ms,mean_rank_average_ms,best_rank_max_ms,worst_rank_max_ms,rms,linf\n");
      }
      for (Method& method : methods) {
        stage(("contract: timed " + method.name).c_str());
        const TimedBlock t = timed_block(method, s.bench.solves, topo.cart_comm());
        // 마지막 풀이의 해로 RMS — 시간 측정 밖
        const fft_test::ContractError e = fft_test::contract_error(method.name == "fftu" ? phi_new : phi_old, grid, sub);
        if (root) {
          std::printf("[MPM-CONTRACT] %-12s %10.4f ms/solve (rank-max)  rms %.10e  expected %.10e  linf %.6e\n",
                      method.name.c_str(), t.mean_rank_max, e.rms, expected_rms, e.linf);
          if (csv)
            std::fprintf(csv, "%s,%.6f,%.6f,%.6f,%.6f,%.10e,%.10e\n", method.name.c_str(), t.mean_rank_max,
                         t.mean_rank_average, t.best_rank_max, t.worst_rank_max, e.rms, e.linf);
        }
        if (s.stretch.uniform_z) {
          CHECK(std::abs(e.rms - expected_rms) < 1e-3 * expected_rms);
        } else {
          CHECK(e.rms < 1e-2);
        }
      }
      if (csv) std::fclose(csv);
      if (s.bench.phases) {
        for (Method& method : methods) {
          stage(("contract: phases " + method.name).c_str());
          mpmstd::common::PhaseTimer timer;
          method.attach_timer(&timer);
          const double tdma_start = method.tdma_comm_seconds();
          const double transpose_start = method.transpose_comm_seconds();
          timer.activate();
          for (int n = 0; n < s.bench.solves; ++n) {
            timer.begin("solve_total");
            method.solve();
            timer.end("solve_total");
            timer.count_step();
          }
          method.attach_timer(nullptr);
          timer.set_external("comm.tdma_z", method.tdma_comm_seconds() - tdma_start);
          timer.set_external("comm.transpose", method.transpose_comm_seconds() - transpose_start);
          if (root) std::printf("# ---- phases: %s ----\n", method.name.c_str());
          timer.report(topo.cart_comm(), prefix.empty() ? "" : prefix + "_" + method.name + "_phases.csv", header);
        }
      }
    } else if (s.bench.solves > 0) {
      stage("bench warm-up");
      fft_test::fill_random_rhs(rhs, grid, sub);
      for (int w = 0; w < s.bench.warmup; ++w) {
        mpmstd::equation::pressure::solve(old_solver, grid, sub, rhs, phi_old);
        new_solver.solve(rhs, phi_new);
      }
      const auto run_pass = [&](const bool phases, const std::string& csv_path) {
        mpmstd::common::PhaseTimer timer;
        old_solver.timer = phases ? &timer : nullptr;
        new_solver.timer = phases ? &timer : nullptr;
        // 누적 MPI 시간 — 이 패스의 증가분만 구간처럼 싣는다 (plot_events.py 가 계산·통신을 가른다)
        const double old_tdma_start = old_solver.plan_z.comm_seconds;
        const double old_transpose_start = old_transpose_comm_seconds(old_solver);
        const double new_tdma_start = new_solver.tdma_comm_seconds();
        stage(phases ? "bench pass: phases" : "bench pass: total");
        timer.activate();
        for (int n = 0; n < s.bench.solves; ++n) {
          timer.begin("old.solve_total");
          mpmstd::equation::pressure::solve(old_solver, grid, sub, rhs, phi_old);
          timer.end("old.solve_total");
          timer.begin("new.solve_total");
          new_solver.solve(rhs, phi_new);
          timer.end("new.solve_total");
          timer.count_step();
        }
        old_solver.timer = nullptr;
        new_solver.timer = nullptr;
        if (phases) {
          timer.set_external("old.comm.tdma_z", old_solver.plan_z.comm_seconds - old_tdma_start);
          timer.set_external("old.comm.transpose", old_transpose_comm_seconds(old_solver) - old_transpose_start);
          timer.set_external("fftu.comm.tdma_z", new_solver.tdma_comm_seconds() - new_tdma_start);
        }
        if (root) std::printf("# ---- bench pass: %s (%d solves each) ----\n", phases ? "phases" : "total",
                              s.bench.solves);
        timer.report(topo.cart_comm(), csv_path, header);
      };
      run_pass(false, prefix.empty() ? "" : prefix + "_total.csv");
      if (s.bench.phases) run_pass(true, prefix.empty() ? "" : prefix + "_phases.csv");
    }
    fft_test::release_transpose_comms();
    return root ? mpmstd::test::summary("FFT_test") : (mpmstd::test::failures == 0 ? 0 : 1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FFT_test rank %d: %s\n", mpi.world_rank(), e.what());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }
}
