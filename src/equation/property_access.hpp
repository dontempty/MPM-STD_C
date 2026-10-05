#pragma once
// 상수물성 접근자 (plan §18-1) — 가변 스텐실의 물성 배열 자리(const real_t*)에
// 그대로 끼우는 영비용 대체물. 물성 배열을 **만들지도 참조하지도 않는** 상수
// 물성 경로가 이걸로 성립한다:
//   UnitArray : μ̂ · 1/ρ̂ · κ̂ · 1/(ρ̂Ĉp) ≡ 1   (어떤 인덱스로 읽어도 1)
//   ZeroArray : 물성 미분장 (dκ̂/dt 등)  ≡ 0
// 산술 구조가 가변 경로와 동일하므로(로드가 리터럴로 바뀔 뿐) 결과는
// "가변 커널 + 1 로 채운 배열" 과 비트 일치해야 한다 — 그게 검증 게이트다.
#include "common/hd.hpp"
#include "common/types.hpp"

namespace mpmstd::equation {

struct UnitArray {
  MPMSTD_HD real_t operator[](long long) const { return real_t(1); }
};
struct ZeroArray {
  MPMSTD_HD real_t operator[](long long) const { return real_t(0); }
};

}  // namespace mpmstd::equation
