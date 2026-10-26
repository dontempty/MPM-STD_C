#pragma once
// [최적화 2026-10-23] 솔버 공용 작업 영역 — docs/Memory/plan.md.
//
// scalar·momentum·pressure 가 각자 들고 있던 장(field) 크기 작업 버퍼(밴드 a·b·c·d·d2, 압력의
// 라인·스펙트럼·FFT 작업 영역)를 앱이 **한 번** 만든 슬롯 5개로 모으고 세 솔버가 빌려 쓴다.
// 한 스텝 안에서 세 풀이는 차례로 돌고(같은 스트림), 솔버는 호출 사이에 슬롯 내용이 남아 있다고
// 가정하지 않으므로(매 호출 조립해서 쓰고 버림) 공유해도 수치는 비트 동일이다.
//
// 슬롯 배정 (plan.md §2.3):
//   scalar / momentum(성분마다) : 0 = A, 1 = B, 2 = C, 3 = D0, 4 = D1(핑퐁)
//   pressure                   : 0 = real_lines ↔ band_c, 1 = spectrum_x, 2 = spectrum_xy(= TDMA 의 D),
//                                3 = band_a ↔ cuFFT 작업 영역, 4 = band_b   (↔ = 수명이 겹치지 않는 쌍)
//
// 불변 조건:
//   * 세 솔버의 solve 는 겹치지 않고 차례로 불린다 — 다른 스트림에서 겹쳐 돌리려면 Workspace 를 나눠야 한다.
//   * 생성 뒤 주소 고정(재할당 없음) — CPU 빌드의 FFTW 플랜이 슬롯 주소에 묶인다.
//   * 통계·진단은 슬롯을 읽지 않는다.
//
// 크기: 앱이 각 솔버의 required_slot_elements(...) 최대값으로 만든다. 솔버 생성자가 모자라면 throw.
//   mpmstd::equation::Workspace ws(std::max({momentum::Solver::required_slot_elements(sub),
//                                            scalar::Solver::required_slot_elements(sub),
//                                            pressure::Solver::required_slot_elements(grid, sub)}));
#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "common/types.hpp"
#include "field/buffer.hpp"

namespace mpmstd::equation {

class Workspace {
 public:
  static constexpr int kSlots = 5;

  explicit Workspace(const std::size_t slot_elements) : slot_elements_(slot_elements) {
    if (slot_elements == 0) throw std::invalid_argument("Workspace: slot_elements must be > 0");
    for (auto& slot : slots_) slot.assign(slot_elements);
  }
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  real_t* slot(const int k) {
    if (k < 0 || k >= kSlots) throw std::out_of_range("Workspace::slot: " + std::to_string(k));
    return slots_[k].data();
  }
  std::size_t slot_elements() const { return slot_elements_; }

  // 솔버 생성자용: 요구량이 슬롯보다 크면 즉시 중단 (조용한 범위 밖 접근 금지).
  void require(const std::size_t need, const char* who) const {
    if (need > slot_elements_)
      throw std::invalid_argument(std::string(who) + ": Workspace slot too small (need "
                                  + std::to_string(need) + ", have "
                                  + std::to_string(slot_elements_) + ")");
  }

 private:
  std::size_t slot_elements_ = 0;
  std::array<field::Buffer, kSlots> slots_;
};

}  // namespace mpmstd::equation
