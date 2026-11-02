#pragma once
// TDMA 중간계층 — 방정식 코드(equation/*)가 보는 유일한 TDMA 헤더.
//
// 세 개의 갈림이 이 파일 하나에 모인다:
// 1) 백엔드 솔버 (2026-10-04 사용자 결정 → 2026-10-28 Filtered 추가): 컴파일 스위치
//    MPMSTD_TDMA_FILTERED (make TDMA=filtered) 하나로 정한다. 입력 키가 아니라 빌드가
//    정한다 (조용한 전환 금지) — 실행 로그 맨 앞에 tdma_backend_name() 을 찍는다.
//    * 기본: PaScaL_TDMA (정확해, 직렬 Thomas 와 같은 해)
//    * filtered: Filtered_TDMA (/scratch/t639a05/Filtered_TDMA libs/filtered_tdma,
//      커밋 11a2c2e 를 수정 없이 vendor) — 인접 랭크 2×2 인터페이스 + J 행 절단 근사.
//      절단 허용오차 eps = dt·Δ̄² (Δ̄ = 축 길이/전역 셀 수, 스트레칭 축도 평균 간격) 를
//      소비자가 매 풀이 전에 set_tolerance 로 넣는다 (Filtered README §9-2 의 논문 기준).
// 2) 정확해 고정 플랜 ExactTdmaPlan: 압력 Poisson 의 z 모드 시스템은 (0,0) 모드가
//    대각지배가 아니라(ρ = 1/2, 감쇠 없음) 절단 근사의 전제가 깨진다 → 백엔드와 무관하게
//    항상 PaScaL. Filtered_TDMA 의 채널 앱도 같은 선택(압력 z 는 PaScaL 고정).
// 3) CPU/GPU (MPMSTD_GPU — 프로젝트 유일의 CPU/GPU 분기 스위치, D5):
//    * CPU: PaScaL_TDMA/pascal_tdma_cpu.*, Filtered_TDMA/cpu/ (FilteredTDMA)
//    * GPU: PaScaL_TDMA/pascal_tdma_cuda.*, Filtered_TDMA/gpu/ (FilteredTDMACUDA)
//      — 밴드·해 포인터는 전부 **디바이스** 포인터다.
//
// 계약(레이아웃 [sys + row*nsys], in-place, 해는 d, 대각지배 전제, 비주기는
// 양끝 a/c = 0)은 CPU 백엔드 헤더(PaScaL_TDMA/pascal_tdma_cpu.hpp)가 정본이다.
// Filtered 의 레이아웃(row-major [n_row × n_sys])과 비주기 끝 처리(이웃 = MPI_PROC_NULL
// 이면 그 인터페이스 보정 생략)는 이 계약과 같다.

#include <mpi.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/types.hpp"

#ifdef MPMSTD_GPU
#include "linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cuda.hpp"
#else
#include "linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cpu.hpp"
#endif

#ifdef MPMSTD_TDMA_FILTERED
#ifdef MPMSTD_GPU
#include "linear_solver/tdma/Filtered_TDMA/include/filtered_tdma_cuda.hpp"
#else
#include "linear_solver/tdma/Filtered_TDMA/include/filtered_tdma.hpp"
#endif
#endif

namespace mpmstd::linear_solver {

// ── 절단 깊이 기록 (filtered 빌드 전용; 2026-10-28) ──
// 플랜의 record_truncation 이 켜져 있으면 풀이마다 {J, fallback} 하나를 truncation 에 쌓는다.
// J 를 읽으려면 디바이스→호스트 복사(동기화)가 필요하므로 앱은 모니터 스텝에만 켠다.
//   J = −1: 그 축이 랭크 안에서 완결(nprocs = 1) → Filtered 도 직렬 Thomas, J 를 계산하지 않음.
//   fallback = 1: 허용오차를 맞추려면 J > n_row−2 가 필요했다(라이브러리는 n_row−2 로 잘라 풀었음).
// PaScaL 빌드에서는 멤버만 있고 비어 있다 (tdma_reports_truncation = false).
struct TruncationSample {
  int J = -1;
  int fallback = 0;
};

// ── 정확해 플랜 (PaScaL) — 라인 컴당 하나 만들어 매 스텝 재사용 (plan.create(nsys, comm)) ──
// 백엔드 플랜의 별칭(using)이 아니라 이 계층 소유의 타입이다 — 별칭이면 인자
// 타입을 따라가는 이름 찾기(ADL)가 백엔드 함수를 직접 집어 중간계층과
// 충돌/우회한다.
struct ExactTdmaPlan {
#ifdef MPMSTD_GPU
  pascal_tdma::PascalTdmaPlan backend;
#else
  pascal_tdma::TdmaPlan backend;
#endif
  // 이 플랜의 solve 가 MPI(Alltoallv) 안에서 보낸 누적 초 — GPU 백엔드의 comm_seconds 훅
  // (측정 오버헤드 없음: 알고리즘이 이미 그 자리에서 스트림을 동기화한다).
  // 확장성 측정(docs/Scalability/plan.md §2.2)이 읽는다. 랭크 내 플랜(x·y)은 nprocs 1 이라 ~0.
  // CPU 백엔드는 아직 재지 않는다 → 0.
  double comm_seconds = 0;

  void create(const int nsys, MPI_Comm line_comm) { backend.create(nsys, line_comm); }
  void destroy() noexcept { backend.destroy(); }
};

// 비주기 (벽 축 — fold 가 끝난 행렬). 해는 d 에. GPU 는 기본 스트림에서 동기적으로 푼다.
inline void solve_many(ExactTdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                       const int nsys, const int nrow) {
#ifdef MPMSTD_GPU
  pascal_tdma::solve(plan.backend, a, b, c, d, nsys, nrow, nullptr, &plan.comm_seconds);
#else
  pascal_tdma::solve_many(plan.backend, a, b, c, d, nsys, nrow);
#endif
}

// 주기 (주기 축 — 첫/끝 행이 랩 계수를 가진 행렬). 해는 d 에.
inline void solve_cyclic_many(ExactTdmaPlan& plan, real_t* a, real_t* b, real_t* c,
                              real_t* d, const int nsys, const int nrow) {
#ifdef MPMSTD_GPU
  pascal_tdma::solve_cycle(plan.backend, a, b, c, d, nsys, nrow, nullptr, &plan.comm_seconds);
#else
  pascal_tdma::solve_cyclic_many(plan.backend, a, b, c, d, nsys, nrow);
#endif
}

#ifndef MPMSTD_TDMA_FILTERED

inline const char* tdma_backend_name() { return "pascal"; }

// ── 운동량·스칼라 플랜 (기본 빌드 = PaScaL) ──
// 정확해 플랜에 절단 허용오차 setter(무시)만 더한다 — 소비자 코드가 백엔드와 무관하게 같다.
// solve_many/solve_cyclic_many 는 기반 타입(ExactTdmaPlan) 판을 그대로 쓴다.
struct TdmaPlan : ExactTdmaPlan {
  void set_tolerance(double) {}  // PaScaL 은 정확해 — 허용오차 없음
  bool record_truncation = false;            // 효과 없음 (절단이 없다)
  std::vector<TruncationSample> truncation;  // 항상 비어 있다
};
inline constexpr bool tdma_reports_truncation = false;

#else  // MPMSTD_TDMA_FILTERED

// 진단용: 환경변수 MPMSTD_FTDMA_EPS_SCALE 가 있으면 eps 에 곱한다 (작게 주면 Filtered → 정확해에 수렴해야 한다 —
// 구현 검증용; docs/Design/20261028_filtered_tdma.md). 쓰면 로그 첫 줄의 백엔드 이름에 배율이 찍힌다.
inline double filtered_eps_scale() {
  static const double scale = [] {
    const char* v = std::getenv("MPMSTD_FTDMA_EPS_SCALE");
    return v ? std::atof(v) : 1.0;
  }();
  return scale;
}

inline const char* tdma_backend_name() {
  static const std::string name = filtered_eps_scale() == 1.0
      ? std::string("filtered")
      : "filtered(eps x " + std::to_string(filtered_eps_scale()) + ", MPMSTD_FTDMA_EPS_SCALE)";
  return name.c_str();
}

#ifdef MPMSTD_GPU
using FilteredBackend = ::FilteredTDMACUDA;
#else
using FilteredBackend = ::FilteredTDMA;
#endif

// ── 운동량·스칼라 플랜 (filtered 빌드) ──
// Filtered 는 n_row 와 좌우 이웃을 생성자에서 고정한다 — create(nsys, comm) 때는 n_row 를
// 모르므로 첫 풀이에서 만든다. 비주기(끝 랭크 이웃 = MPI_PROC_NULL)와 주기(랩) 는 이웃이
// 달라 객체를 따로 둔다. 한 플랜 = 한 축이라 실제로는 둘 중 하나만 생긴다.
struct TdmaPlan {
  int nsys = 0;
  int nrow = 0;  // 첫 풀이에서 고정 — 이후 다른 nrow 가 오면 throw (계약 위반)
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = 0, nprocs = 1;
  double tolerance = 0;  // eps = dt·Δ̄² — 풀이 전에 set_tolerance 로 넣는다 (0 이면 throw)
  std::unique_ptr<FilteredBackend> plain, cyclic;
  double comm_seconds = 0;  // Filtered 의 비계측 solve 는 통신 시간을 재지 않는다 → 0
  bool record_truncation = false;            // 켜면 풀이마다 {J, fallback} 를 쌓는다 (동기화 비용)
  std::vector<TruncationSample> truncation;  // 앱이 읽고 비운다

  void create(const int nsys_in, MPI_Comm line_comm) {
    nsys = nsys_in;
    comm = line_comm;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
  }
  void destroy() noexcept {
    plain.reset();
    cyclic.reset();
  }
  void set_tolerance(const double eps_in) {
    const double eps = eps_in * filtered_eps_scale();
    tolerance = eps;
    if (plain) plain->set_eps(eps);
    if (cyclic) cyclic->set_eps(eps);
  }
};

namespace detail {

inline FilteredBackend& filtered_backend(TdmaPlan& plan, const int nsys, const int nrow,
                                         const bool periodic) {
  if (nsys != plan.nsys)
    throw std::invalid_argument("TdmaPlan(filtered): nsys " + std::to_string(nsys) +
                                " != plan nsys " + std::to_string(plan.nsys));
  if (plan.nrow == 0) plan.nrow = nrow;
  if (nrow != plan.nrow)
    throw std::invalid_argument("TdmaPlan(filtered): nrow " + std::to_string(nrow) +
                                " != first solve's nrow " + std::to_string(plan.nrow));
  // 커널이 행 J+1·(n_row−2)−J 를 짚는다 (0 ≤ J ≤ n_row−2) — 분산 풀이는 3 행 이상 필요
  if (plan.nprocs > 1 && nrow < 3)
    throw std::invalid_argument("TdmaPlan(filtered): distributed solve needs >= 3 rows per rank, got " +
                                std::to_string(nrow));
  if (!(plan.tolerance > 0))
    throw std::logic_error("TdmaPlan(filtered): set_tolerance(dt*dx^2) must be called before solving");
  std::unique_ptr<FilteredBackend>& slot = periodic ? plan.cyclic : plan.plain;
  if (!slot) {
    const int n = plan.nprocs, r = plan.rank;
    const int left = periodic ? (r - 1 + n) % n : (r > 0 ? r - 1 : MPI_PROC_NULL);
    const int right = periodic ? (r + 1) % n : (r < n - 1 ? r + 1 : MPI_PROC_NULL);
    slot = std::make_unique<FilteredBackend>(nsys, nrow, r, n, plan.comm, left, right,
                                             plan.tolerance);
  }
  return *slot;
}

}  // namespace detail

inline constexpr bool tdma_reports_truncation = true;

namespace detail {
inline void record(TdmaPlan& plan, const FilteredBackend& backend) {
  if (!plan.record_truncation) return;
  TruncationSample sample;  // nprocs = 1 → 직렬 폴백, J 미계산 (−1)
  if (plan.nprocs > 1) {
    sample.J = backend.last_J();
    sample.fallback = backend.last_fallback() ? 1 : 0;
  }
  plan.truncation.push_back(sample);
}
}  // namespace detail

inline void solve_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                       const int nsys, const int nrow) {
  FilteredBackend& backend = detail::filtered_backend(plan, nsys, nrow, false);
  backend.solve(a, b, c, d);
  detail::record(plan, backend);
}

inline void solve_cyclic_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c,
                              real_t* d, const int nsys, const int nrow) {
  FilteredBackend& backend = detail::filtered_backend(plan, nsys, nrow, true);
  backend.solve_cyclic(a, b, c, d);
  detail::record(plan, backend);
}

#endif  // MPMSTD_TDMA_FILTERED

// 절단 허용오차 eps = dt·Δ̄² (Filtered 논문 기준; Δ̄ = 축 전역 길이 / 전역 셀 수).
// 축마다 하나 — 운동량·스칼라 ADI 가 스윕 전에 부른다. PaScaL 빌드에서는 효과 없음.
inline double truncation_tolerance(const real_t dt, const real_t axis_length, const int n_global) {
  const double mean_spacing = static_cast<double>(axis_length) / n_global;
  return static_cast<double>(dt) * mean_spacing * mean_spacing;
}

}  // namespace mpmstd::linear_solver
