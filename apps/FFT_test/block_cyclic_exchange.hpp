#pragma once
// 장(블록 분배, ghost 포함) ↔ y 순환 x 라인 — FFTU 의 앞뒤에 붙는 재분배 (plan.md §3.1).
//
//   to_cyclic : rhs 내부 → lines[(k·nz + z)·nx + x], k = y div q                 fftu.block_to_cyclic
//   to_block  : lines · scale − subtract → phi 내부 (ghost 는 건드리지 않는다)        fftu.cyclic_to_block
//
// 인덱스 표 없이 기존 평면 전치(pencil_exchange.cu)와 같은 구조로 간다:
//   블록 쪽: 내 블록의 y ≡ g (mod q) 평면들 → 버퍼 [(i·nz + z)·bxn + xl] — 32×32 타일 전치 커널
//   라인 쪽: 상대 r 의 조각은 lines 의 연속 행 [k0·nz + (i·nz + z)] 의 x 구간 — 행 복사
// q = 1 이면 MPI 도 버퍼도 없이 장 ↔ lines 를 바로 옮긴다 (기존 슬랩 경로의 팩·산포와 같다).
// 버퍼는 CyclicFft 의 슬롯을 빌린다 (block_scratch, cyclic_scratch).

#include <vector>

#include "common/phase_timer.hpp"
#include "cyclic_layout.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace fft_test {

class BlockCyclicExchange {
 public:
  BlockCyclicExchange(const mpmstd::parallel::Subdomain& sub, const CyclicLayout& layout, bool host_staging);

  void to_cyclic(const mpmstd::field::Field& rhs, real_t* lines, real_t* block_scratch, real_t* cyclic_scratch,
                 mpmstd::common::PhaseTimer* timer);
  void to_block(const real_t* lines, mpmstd::field::Field& phi, real_t scale, real_t subtract, real_t* block_scratch,
                real_t* cyclic_scratch, mpmstd::common::PhaseTimer* timer);

 private:
  const CyclicLayout& layout_;
  bool host_staging_ = false;
  int ny_total_ = 0, nz_total_ = 0;
  int block_x_count_ = 0;
  // 블록 쪽 — 그룹 랭크 g 에게 갈 내 평면: 로컬 y (ghost 제외) first_local_y_[g] + i·q, i < plane_count_[g]
  std::vector<int> first_local_y_, plane_count_;
  // 라인 쪽 — 상대 r 의 조각: x 구간 [peer_x_offset_[r], +peer_x_count_[r]), 행 peer_first_k_[r]·nz 부터 peer_plane_count_[r]·nz 행
  std::vector<int> peer_x_offset_, peer_x_count_, peer_first_k_, peer_plane_count_;
  // MPI (double 단위; 블록 → 순환 기준. to_block 은 반대로 쓴다)
  std::vector<int> send_counts_, send_displs_, recv_counts_, recv_displs_;
};

}  // namespace fft_test
