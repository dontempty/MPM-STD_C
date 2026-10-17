#pragma once
// [최적화 2026-10-18] 밴드 핑퐁 — 다음 스윕의 조립이 직전 TDMA 결과 밴드를 직접 읽어 scatter_x/y(밴드 → 필드 복사)를 없앤다.
//   Field = 전체 모양 필드, BandX = x 밴드 (ii·i2 + jj)·i3 + kk, BandY = y 밴드 (jj·i1 + ii)·i3 + kk (둘 다 k 연속 → 병합 읽기).
//   z 밴드는 전치 배치라 제외(scatter_z 가 타일 전치). 비트 동일.
#include <cstddef>

#include "common/hd.hpp"

namespace mpmstd::equation {

enum class SourceLayout : int { Field = 0, BandX = 1, BandY = 2 };

MPMSTD_HD inline std::size_t source_index(const SourceLayout layout, const int ii, const int jj,
                                          const int kk, const int i1, const int i2, const int i3,
                                          const std::size_t field_index) {
  if (layout == SourceLayout::BandX)
    return (static_cast<std::size_t>(ii) * i2 + jj) * i3 + kk;
  if (layout == SourceLayout::BandY)
    return (static_cast<std::size_t>(jj) * i1 + ii) * i3 + kk;
  return field_index;
}

}  // namespace mpmstd::equation
