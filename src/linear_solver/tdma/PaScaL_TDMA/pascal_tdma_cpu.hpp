#pragma once
// PaScaL_TDMA — 분할(분산) 삼중대각 솔버의 CPU 판.
//
// ── 이 폴더가 "PaScaL_TDMA 의 집"이다 (2026-10-04 사용자 결정) ─────────────
// 솔버의 정의는 전부 이 폴더 안에 있다:
//   pascal_tdma_cuda.{hpp,cu} — 받아온 C++/CUDA 포트 (실전 GPU).
//                               cyclic 없음 + 랭크당 2행 버그 → M6-GPU 숙제.
//   pascal_tdma_cpu.{hpp,cpp} — 이 파일: CPU 판 (스모크·검증용; cyclic 포함).
//   *.f90                     — 쓰지 않는다 (사용자 결정).
// 소비자(equation/*)는 이 헤더를 직접 보지 않고 **중간계층
// `linear_solver/tdma/tdma.hpp`** 를 include 한다 — 나중에 Filtered_TDMA 를
// 들여올 때(폴더 `../Filtered_TDMA/` + 네임스페이스 filtered_tdma 로 옆에 추가)
// 바꾸는 곳이 그 파일 하나가 되도록. 이 헤더를 직접 include 하는 곳은 백엔드
// 내부(직렬 커널)를 검사하는 테스트뿐이다.
// ⚠ Filtered 를 가져오는 날: 연구실 CPU 판은 np≥2 발산 이력이 있다 — GPU 판을
//   기준으로 이식할 것.
//
// ── 배열 레이아웃 (PaScaL_TDMA 와 동일) ─────────────────────────────────────
//   계수[sys + row * nsys]   — sys 가 메모리에서 가장 빠르다.
//   row  = 풀이 방향의 행 번호 (이 랭크 로컬, ghost 제외)
//   sys  = 서로 독립인 시스템 번호 (예: z-스윕이면 (i,j) 평면의 한 점)
// 주의: 필드 배열은 k 가 가장 빠르므로((i*ny+j)*nz+k), z-스윕의 계수·RHS 는
// 조립 단계(M7+)에서 이 레이아웃으로 "옮겨 담으며" 만든다. x-스윕은 반대로
// 필드 레이아웃과 자연히 일치한다. 어느 쪽이든 조립이 책임진다 — 솔버는
// 위 레이아웃 하나만 안다.
//
// ── 계약 ────────────────────────────────────────────────────────────────────
// * in-place: a, b, c, d 네 배열 모두 파괴된다. 해는 d 에 담겨 나온다.
// * 피벗 없음 — 대각 지배 행렬 전제 (확산 음해항이 항상 만족).
// * 비주기(solve_many): 전역 첫 행의 a 와 전역 끝 행의 c 는 0 이어야 한다
//   (벽 쪽은 fold 가 이미 소거했다 — docs/conventions.md §6).
// * 주기(solve_cyclic_many): 전역 첫 행의 a 는 x[전역 끝], 전역 끝 행의 c 는
//   x[전역 첫] 에 걸리는 랩 계수다.
//
// ── 분할 알고리즘 (PaScaL_TDMA; 축약계 접근) ────────────────────────────────
// 1) 각 랭크가 자기 nrow 행을 "수정 Thomas"로 소거 → 내부 행들이 로컬 양끝
//    미지수(x_first, x_last)만의 함수가 되고, 양끝 행 2개가 랭크 간 결합을 들고
//    남는다.
// 2) 전 랭크의 양끝 행(랭크당 2행)을 모으면 크기 2*nprocs 의 작은 삼중대각계.
//    시스템들을 랭크에 나눠 (alltoall 전치) 각자 자기 몫을 푼다.
//    ★ 주기 문제는 여기만 다르다: 이 축약계가 "순환" 삼중대각이 되므로
//    Thomas 대신 순환 Thomas 로 푼다. 로컬 소거(1)와 되살리기(3)는 동일.
// 3) 양끝 해를 돌려받아(역 alltoall) 내부 행을 되살린다:
//    x_j = d_j − a_j·x_first − c_j·x_last.
//
// ── 제약 ────────────────────────────────────────────────────────────────────
// * 라인 컴 안의 모든 랭크가 같은 nsys 를 가져야 한다 (create 가 검사해서
//   throw). 데카르트 토폴로지의 한 축 컴은 자동으로 만족한다 — 같은 라인의
//   랭크들은 나머지 두 축 분할이 동일하므로.
// * nprocs ≥ 2 면 랭크당 nrow ≥ 2 필요 (양끝 행이 축약계의 행이 되므로).
// * nsys < nprocs 는 지원하지 않는다 (split_evenly 가 throw — 빈 랭크 금지).

#include <mpi.h>

#include <vector>

#include "common/types.hpp"

namespace mpmstd::pascal_tdma {

// ── 직렬 커널 (한 랭크 안에서 nsys 개 시스템을 독립적으로) ──────────────────

// 보통 Thomas. nrow ≥ 1.
void thomas_many(real_t* a, real_t* b, real_t* c, real_t* d, int nsys, int nrow);

// 순환(주기) Thomas — Sherman–Morrison 보정(랩을 rank-1 수정으로 벗겨 보통
// Thomas 두 RHS 로 풀고 되섞기). nrow ≥ 3 이 본 경로, 1·2 는 특수식.
// nrow == 1 은 자기 자신으로 랩: (a+b+c)·x = d — 2차원 축퇴(ny=1 주기 y)가
// 실제로 이 경로를 쓴다.
void thomas_cyclic_many(real_t* a, real_t* b, real_t* c, real_t* d, int nsys, int nrow);

// ── 분할 플랜 — row 방향이 여러 랭크에 걸쳐 있을 때 ─────────────────────────
// 통신 상대·크기와 작업 버퍼를 한 번 만들어 매 스텝 재사용한다.
// 같은 플랜으로 solve_many / solve_cyclic_many 둘 다 부를 수 있다
// (로컬 소거가 동일하므로 — 위 §2 참조).
struct TdmaPlan {
  MPI_Comm comm = MPI_COMM_NULL;  // dup 사본 — destroy 가 해제한다
  int rank = 0;
  int nprocs = 1;
  int nsys = 0;  // 이 랭크의 시스템 수 (라인 컴 전 랭크 동일 — create 가 검사)

  // 축약계 전치: 시스템들을 랭크에 나눈 몫 (split_evenly 와 동일 규칙).
  std::vector<int> chunk_first;  // 랭크 r 이 푸는 시스템 구간 시작
  std::vector<int> chunk_count;  // 랭크 r 이 푸는 시스템 수
  int my_chunk_count = 0;        // = chunk_count[rank]

  // alltoallv 카운트/변위 (원소 단위, 전진 = 양끝 행 3배열, 후진 = 해 1배열)
  std::vector<int> forward_send_counts, forward_send_displs;
  std::vector<int> forward_recv_counts, forward_recv_displs;
  std::vector<int> backward_send_counts, backward_send_displs;
  std::vector<int> backward_recv_counts, backward_recv_displs;

  // 작업 버퍼 (매 solve 재사용; 크기는 create 가 잡는다)
  std::vector<real_t> edge_a, edge_c, edge_d;  // 내 양끝 행: [sys + row2*nsys], row2 ∈ {0(첫),1(끝)}
  std::vector<real_t> reduced_a, reduced_b, reduced_c, reduced_d;  // 내 몫 축약계: [sys + row*my_chunk_count], row < 2*nprocs
  std::vector<real_t> send_buffer, recv_buffer;
  // 순환 Thomas 의 Sherman–Morrison 스크래치 — solve_cyclic_many 가 첫 호출 때
  // 크기를 잡고 재사용한다 (GPU managed memory 에서 매 호출 할당은 ms 급이라 금물)
  std::vector<real_t> cyclic_scratch;

  bool created = false;

  TdmaPlan() = default;
  ~TdmaPlan() { destroy(); }
  TdmaPlan(const TdmaPlan&) = delete;
  TdmaPlan& operator=(const TdmaPlan&) = delete;

  // line_comm: 풀이 방향 한 줄의 컴 (예: topo.z().comm). nprocs == 1 도 된다.
  void create(int nsys_in, MPI_Comm line_comm);
  void destroy() noexcept;
};

// 비주기 (벽 축 — fold 가 끝난 행렬). 해는 d 에.
void solve_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                int nsys, int nrow);

// 주기 (주기 축 — 첫/끝 행이 랩 계수를 가진 행렬). 해는 d 에.
void solve_cyclic_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                       int nsys, int nrow);

}  // namespace mpmstd::pascal_tdma
