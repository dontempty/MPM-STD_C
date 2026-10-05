#pragma once
// 압력 투영의 보정(correction) 단계 — Poisson solve(solve.hpp)와 분리된 모듈.
//   δp 를 푼 다음에 하는 일들: 속도 보정, P 누적, (NOB) 예측자 전진.
// 상수/가변 물성 축은 다른 모듈과 같은 접미 규칙(*_constant_property /
// *_variable_property); accumulate·advance_pressure_predictor 는 물성 무관 공용.

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::pressure {

// u ← u* − dt·∇φ (이 랭크 소유 면만; staggered 기울기 = Δφ/중심간거리).
// 하단 벽면 W 는 Neumann ghost 덕에 기울기가 정확히 0 — 자동으로 보존된다.
// 호출 뒤 U, V, W 의 ghost 는 낡는다 — 다음 사용 전 교환 + 벽 처리는 호출자 몫.
// (구현은 solve_kernels.{cpp,cu} — 백엔드별)
void correct_velocity_constant_property(field::Field& U, field::Field& V, field::Field& W,
                      const field::Field& phi, const grid::Grid& grid,
                      const parallel::Subdomain& sub, real_t dt);

// P += φ — ghost 포함 전체 칸. 둘 다 ghost 일관 상태라 덧셈만으로 P 의 ghost 도
// 유지된다 (재교환 불필요 — MPM-STD_C 의 pre.f90:818 관례 그대로).
void accumulate(field::Field& P, const field::Field& phi);

// ── 가변 밀도(NOB) 투영 (PaScaL_TCS §804-821 1:1) ───────────────────────────
// u −= dt·[ ∇(δp) + ((1/ρ̂)_면 − 1)·∇p̂ ].  rhs 의 ExtraTerm 과 짝 —
// 합치면 가변 밀도 투영이 반복 없이 닫힌다. p̂(dPhat)·1/ρ̂ ghost 최신 전제.
void correct_velocity_variable_property(field::Field& U, field::Field& V, field::Field& W,
                               const field::Field& phi,
                               const field::Field& pressure_predictor,
                               const field::Field& inverse_density,
                               const grid::Grid& grid, const parallel::Subdomain& sub,
                               real_t dt);

// 다음 스텝용 예측자 2차 외삽 (ghost 포함 전체; 호출 뒤 p̂ 를 교환+벽 채움):
//   previous ← (p̂ + previous)/2,   p̂ ← 2·δp − previous
void advance_pressure_predictor(field::Field& pressure_predictor,
                                field::Field& previous_predictor,
                                const field::Field& phi);

}  // namespace mpmstd::equation::pressure
