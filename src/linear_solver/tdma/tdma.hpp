#pragma once
// TDMA 중간계층 — 방정식 코드(equation/*)가 보는 유일한 TDMA 헤더.
//
// 두 개의 갈림이 이 파일 하나에 모인다:
// 1) 백엔드 솔버 교체 (2026-10-04 사용자 결정): 지금은 PaScaL_TDMA 하나지만
//    나중에 Filtered_TDMA 를 추가한다 — 그날 바꾸는 곳은 이 파일뿐이고
//    소비자(equation/*)는 무수정이다. 어느 백엔드인지는 입력 키가 아니라
//    아래 코드가 정한다 (조용한 전환 금지).
// 2) CPU/GPU (MPMSTD_GPU — 프로젝트 유일의 분기 스위치, D5):
//    * CPU: PaScaL_TDMA/pascal_tdma_cpu.*  (우리가 작성한 참조 구현)
//    * GPU: PaScaL_TDMA/pascal_tdma_cuda.* (벤더 CUDA 포트 + 우리가 추가한
//      solve_cycle·nrow==2 수정) — 밴드·해 포인터는 전부 **디바이스** 포인터다.
//
// 계약(레이아웃 [sys + row*nsys], in-place, 해는 d, 대각지배 전제, 비주기는
// 양끝 a/c = 0)은 CPU 백엔드 헤더(PaScaL_TDMA/pascal_tdma_cpu.hpp)가 정본이다.

#ifdef MPMSTD_GPU

#include "linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cuda.hpp"

namespace mpmstd::linear_solver {

// 분할 플랜 — 라인 컴당 하나 만들어 매 스텝 재사용 (plan.create(nsys, comm)).
// 백엔드 플랜을 멤버로 소유한다 (CPU 판과 같은 이유 — ADL 우회 방지 + 교체 지점).
struct TdmaPlan {
  pascal_tdma::PascalTdmaPlan backend;

  void create(const int nsys, MPI_Comm line_comm) { backend.create(nsys, line_comm); }
  void destroy() noexcept { backend.destroy(); }
};

// a, b, c, d 는 디바이스 포인터. 기본 스트림에서 동기적으로 푼다.
inline void solve_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                       const int nsys, const int nrow) {
  pascal_tdma::solve(plan.backend, a, b, c, d, nsys, nrow);
}

inline void solve_cyclic_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c,
                              real_t* d, const int nsys, const int nrow) {
  pascal_tdma::solve_cycle(plan.backend, a, b, c, d, nsys, nrow);
}

}  // namespace mpmstd::linear_solver

#else  // CPU

#include "linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cpu.hpp"

namespace mpmstd::linear_solver {

// 분할 플랜 — 라인 컴당 하나 만들어 매 스텝 재사용 (plan.create(nsys, comm)).
// 백엔드 플랜의 별칭(using)이 아니라 이 계층 소유의 타입이다 — 별칭이면 인자
// 타입을 따라가는 이름 찾기(ADL)가 백엔드 함수를 직접 집어 중간계층과
// 충돌/우회하고, Filtered 가 들어오면 어차피 자기 타입이 필요하다.
struct TdmaPlan {
  pascal_tdma::TdmaPlan backend;  // 지금의 유일한 백엔드

  void create(const int nsys, MPI_Comm line_comm) { backend.create(nsys, line_comm); }
  void destroy() noexcept { backend.destroy(); }
};

// 비주기 (벽 축 — fold 가 끝난 행렬). 해는 d 에.
inline void solve_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                       const int nsys, const int nrow) {
  pascal_tdma::solve_many(plan.backend, a, b, c, d, nsys, nrow);
}

// 주기 (주기 축 — 첫/끝 행이 랩 계수를 가진 행렬). 해는 d 에.
inline void solve_cyclic_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c,
                              real_t* d, const int nsys, const int nrow) {
  pascal_tdma::solve_cyclic_many(plan.backend, a, b, c, d, nsys, nrow);
}

}  // namespace mpmstd::linear_solver

#endif
