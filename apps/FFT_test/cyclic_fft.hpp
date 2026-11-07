#pragma once
// x 실수 FFT + y 순환 FFT (FFTU, Koopman & Bisseling 2023 Alg 2.2·3.1, p1 = 1) — z 장 nz 개를 한꺼번에. 레이아웃은 cyclic_layout.hpp.
//
//   forward (lines → spectrum):
//     x r2c                                   fftu.fft_x_fwd
//     재배열 (기존 reorder_x_to_y 그대로)        fftu.reorder_x_to_y
//     수퍼스텝 0  로컬 y FFT (길이 m)            fftu.fft_y_fwd
//                 트위들 ω_ny^{k·s} + 패킹       fftu.twiddle_pack_fwd
//     수퍼스텝 1  그룹 all-to-all 1 회           fftu.a2a_fwd
//                 언팩 (보낸 랭크를 안쪽 축으로)    fftu.unpack_fwd
//     수퍼스텝 2  작은 q 점 FFT                  fftu.fft_small_fwd
//   backward (spectrum → lines): 위의 정확한 역순. 왕복 = nx·ny 배 (비정규화).
//   q = 1 이면 트위들·패킹·all-to-all·언팩·작은 FFT 를 모두 건너뛴다 (기존 슬랩 경로와 같은 연산).
//   팩·언팩 커널은 전부 쓰는 쪽 연속 순서로 돈다 (보폭 쓰기가 보폭 읽기보다 3 배 느렸다, v1 실측).
//
// 슬롯 4 개를 소유하고, 수명이 겹치지 않는 쌍이 나눠 쓴다 (q > 1):
//   A = lines ↔ recv ↔ band_a,  B = spectrum_x ↔ send ↔ band_b ↔ 재분배 블록 쪽,
//   C = spectrum_y ↔ band_c ↔ 재분배 순환 쪽,  D = spectrum.
//   q = 1 이면 spectrum = C, band_c = D.

#include <memory>
#include <vector>

#include "batched_fft.hpp"
#include "common/phase_timer.hpp"
#include "cyclic_layout.hpp"
#include "field/buffer.hpp"

namespace fft_test {

class CyclicFft {
 public:
  // min_slot_elements: 재분배 scratch 로 쓸 최소 double 수 (이 랭크의 블록 셀 수)
  CyclicFft(const CyclicLayout& layout, std::size_t min_slot_elements);
  CyclicFft(const CyclicFft&) = delete;
  CyclicFft& operator=(const CyclicFft&) = delete;

  void forward(mpmstd::common::PhaseTimer* timer);   // lines → spectrum
  void backward(mpmstd::common::PhaseTimer* timer);  // spectrum → lines

  const CyclicLayout& layout() const { return layout_; }
  bool host_staging() const { return host_staging_; }

  real_t* lines() { return slot_a_.data(); }
  real_t* spectrum() { return spectrum_buffer().data(); }
  real_t* band_a() { return slot_a_.data(); }
  real_t* band_b() { return slot_b_.data(); }
  real_t* band_c() { return layout_.group_size() > 1 ? slot_c_.data() : slot_d_.data(); }
  real_t* block_scratch() { return slot_b_.data(); }
  real_t* cyclic_scratch() { return slot_c_.data(); }
  // 시험용 — 슬롯 전체 크기로 upload/download 한다
  mpmstd::field::Buffer& lines_buffer() { return slot_a_; }
  mpmstd::field::Buffer& spectrum_buffer() { return layout_.group_size() > 1 ? slot_d_ : slot_c_; }

 private:
  const CyclicLayout& layout_;
  bool host_staging_ = false;
  mpmstd::field::Buffer slot_a_, slot_b_, slot_c_, slot_d_;
  mpmstd::field::Buffer twiddle_;        // ω_ny^{k·s}, m 복소
  std::vector<int> counts_, displs_;     // all-to-all (double 단위, 균등)
  std::unique_ptr<BatchedRealFft> fft_x_;
  std::unique_ptr<BatchedComplexFft> fft_y_, fft_small_;
};

}  // namespace fft_test
