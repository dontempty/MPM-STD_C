#pragma once
// main 호출 단위 구간 타이머 — 확장성 측정 (docs/Scalability/plan.md §2).
//
// 쓰는 법 (apps/rbc_nob/main.cpp 가 첫 소비자):
//   PhaseTimer timer;                       // 비활성으로 시작 — begin/end 는 no-op (워밍업)
//   timer.activate();                       // 워밍업 끝 (phase_timing_start_step)
//   timer.begin("momentum_solve"); …; timer.end("momentum_solve");
//   timer.count_step();                     // 스텝 하나가 끝날 때
//   timer.set_external("tdma_comm_momentum_z", seconds);   // 밖에서 누적된 값(TDMA 훅)도 같은 표에
//   timer.report(comm, "…/phase_timing.csv", {{"nx","256"}, …});
//
// 경계마다 GPU 빌드에서는 cudaDeviceSynchronize() 를 먼저 한다 — 비동기 커널을 자기
// 구간에 귀속시키기 위해서다 (커널 간 오버랩은 사라지므로 오버헤드는 on/off 비교로 기록,
// plan.md §2.1). 구간 이름은 첫 begin 순서로 등록되어 표도 그 순서(= 호출 순서)로 나온다.
// 중첩 가능 (step_total 이 전체를 감싼다) — 구간마다 시작 시각을 따로 둔다.
//
// 보고: 구간별 랭크 합을 MPI_Reduce(min/avg/max) 해 루트가 ms/step 표를 찍고 CSV 로 쓴다.
// share = max/step_total(max). 비용 지표는 **랭크 max** (가장 늦은 랭크가 스텝을 정한다).

#include <mpi.h>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

namespace mpmstd::common {

class PhaseTimer {
 public:
  void activate() { active_ = true; }
  bool active() const { return active_; }
  long steps() const { return n_steps_; }

  void begin(const char* name) {
    if (!active_) return;
    const int id = index(name);
    synchronize_device();
    start_[id] = MPI_Wtime();
  }
  void end(const char* name) {
    if (!active_) return;
    const int id = index(name);
    synchronize_device();
    sum_[id] += MPI_Wtime() - start_[id];
  }
  void count_step() {
    if (active_) ++n_steps_;
  }
  // 밖에서 누적한 초(예: TDMA 플랜의 comm_seconds)를 구간처럼 싣는다 — 덮어쓴다.
  void set_external(const char* name, const double seconds) {
    sum_[index(name)] = seconds;
  }

  // 루트가 표를 stdout 에, CSV 를 csv_path 에 쓴다 (csv_path 비면 표만). header 는 CSV 머리의
  // "# k=v" 줄들 (격자·랭크 수 등 — 노트북이 읽는다).
  void report(MPI_Comm comm, const std::string& csv_path,
              const std::vector<std::pair<std::string, std::string>>& header) const {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const int n = static_cast<int>(names_.size());
    std::vector<double> mn(n), mx(n), total(n);
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    if (n > 0) {
      MPI_Reduce(sum_.data(), mn.data(), n, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(sum_.data(), mx.data(), n, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(sum_.data(), total.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    if (rank != 0) return;
    const double steps = n_steps_ > 0 ? static_cast<double>(n_steps_) : 1.0;
    double reference = 0;  // share 의 분모 = step_total 의 max, 없으면 구간 max 의 합
    for (int i = 0; i < n; ++i)
      if (names_[i] == "step_total") reference = mx[i];
    if (reference <= 0)
      for (int i = 0; i < n; ++i) reference += mx[i];
    std::printf("# phase timing: %ld steps, %d ranks (ms/step; share = max/step_total)\n",
                n_steps_, nranks);
    std::printf("# %-24s %10s %10s %10s %8s %9s\n", "phase", "min", "avg", "max", "share%",
                "imbal%");
    FILE* csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
    if (csv) {
      for (const auto& kv : header) std::fprintf(csv, "# %s=%s\n", kv.first.c_str(), kv.second.c_str());
      std::fprintf(csv, "# steps=%ld ranks=%d\n", n_steps_, nranks);
      std::fprintf(csv, "phase,min_ms,avg_ms,max_ms,share,imbalance\n");
    }
    for (int i = 0; i < n; ++i) {
      const double avg = total[i] / nranks;
      const double share = reference > 0 ? mx[i] / reference : 0;
      const double imbalance = avg > 0 ? (mx[i] - mn[i]) / avg : 0;
      std::printf("# %-24s %10.3f %10.3f %10.3f %8.2f %9.2f\n", names_[i].c_str(),
                  1e3 * mn[i] / steps, 1e3 * avg / steps, 1e3 * mx[i] / steps, 100 * share,
                  100 * imbalance);
      if (csv)
        std::fprintf(csv, "%s,%.6f,%.6f,%.6f,%.6f,%.6f\n", names_[i].c_str(), 1e3 * mn[i] / steps,
                     1e3 * avg / steps, 1e3 * mx[i] / steps, share, imbalance);
    }
    if (csv) std::fclose(csv);
    std::fflush(stdout);
  }

 private:
  int index(const char* name) {
    for (std::size_t i = 0; i < names_.size(); ++i)
      if (names_[i] == name) return static_cast<int>(i);
    names_.emplace_back(name);
    sum_.push_back(0);
    start_.push_back(0);
    return static_cast<int>(names_.size()) - 1;
  }
  static void synchronize_device() {
#ifdef MPMSTD_GPU
    cudaDeviceSynchronize();
#endif
  }

  bool active_ = false;
  long n_steps_ = 0;
  std::vector<std::string> names_;
  std::vector<double> sum_, start_;
};

}  // namespace mpmstd::common
