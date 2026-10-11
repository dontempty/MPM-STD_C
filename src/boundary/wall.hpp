#pragma once
// 전역 벽의 ghost 채움 — 경계항 처리의 "명시(explicit) 반쪽".
//
// ── 설계 규칙 ────────────────────────────────────────────────────────────────
// * 정책 = 함수 이름이다 (mirror / value / zero_gradient / zero_faces). 이 층엔
//   enum 이 없다. 입력 [boundary] 테이블 → 함수 선택은 한 층 위 boundary/apply.hpp
//   가 한다 (기존 앱은 main 이 z 함수를 직접 고른다 — 둘 다 허용).
// * 호출 순서 규칙은 하나: **halo 교환 먼저, 벽 ghost 나중.**
//   (교환이 ghost 평면을 덮을 수 있으므로. 주기 축은 교환의 랩이 이미 처리한다.)
// * 각 함수는 벽을 소유한 랭크에서만 동작한다 — 판정은 함수 안에서
//   neighbor == MPI_PROC_NULL 하나로 한다. main 은 랭크 구분 없이 같은 코드를 쓴다.
//
// ── ghost 공식 (벽값 v, 첫 내부값 f₁) ──────────────────────────────────────
//   mirror        : ghost = 2v − f₁   (2차 — 선형 보간하면 벽 면 위에 정확히 v) → T
//   value         : ghost = v         (1차 — mirror 는 ν/dz² 과잉 감쇠로 채널의
//                                      아임계 천이를 죽인 전례가 있어 접선 속도는
//                                      이쪽이 관례)                            → U, V
//   zero_gradient : ghost = f₁        (∂f/∂n = 0)                              → P
//   zero_faces    : 벽 "면" 자체 = 0  (z-면 필드 W; ghost 가 아니라 면)
//
// ⚠ 비영 Neumann 을 추가하는 날: ghost = f₁ ± (중심간거리)·구배 로 격자 간격을
//   반드시 곱할 것 (CaNS set_bc 의 dr 방식). MPM-STD_C 는 안 곱해서 "value 가
//   구배가 아니라 Δφ" 라는 문서화된 함정이 있었다.
//
// ── 시간 가변 벽값 v(t) ─────────────────────────────────────────────────────
// 이 명시 반쪽은 수정이 필요 없다 — 매 스텝 그 시점 값을 넘기면 된다.
// 수정이 필요한 곳은 음해 반쪽(M7 의 행렬 fold)이다: δ-소거에서 2v 가 사라지는
// 것은 vⁿ⁺¹ = vⁿ 일 때뿐이므로, v(t) 가 되면 벽 행 RHS 에 2Δv·A 항을 추가해야
// 한다. 체크리스트는 docs/conventions.md §6. (현 케이스는 전부 상수 벽값.)
//
// ── 축 매개변수화 (2026-10-05 BC 프레임워크, D13 완화) ───────────────────────
// 처음엔 z 전용 함수뿐이었다 (모든 앱의 벽이 z). cavity 의 x 벽을 위해 아래
// "축·면 인자 판" fill_wall_*(f, sub, axis, side, …) 을 본체로 두고, z 전용 이름은 그
// axis=2 호출로 남긴다 (기존 앱 무수정 + 비트 일치 게이트 통과). 어느 축을 어떤
// 정책으로 채울지는 boundary/apply.hpp 가 BoundaryCondition 테이블을 보고 정한다.
// 진짜 본체는 압력 솔버다: x/y 가 벽이면 그 방향 FFT 를 DCT 로 바꿔야 한다
// (pressure/solve — CaNS initsolver 의 BC 조합별 고유값 표 참조).

#include "common/types.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::boundary {

// ── 축·면 인자 판 (본체). axis: 0 = x, 1 = y, 2 = z.  side: 0 = minus 벽, 1 = plus 벽.
// 한 호출 = 한 벽면. 그 벽을 소유하지 않은 랭크에선 no-op (이웃 == MPI_PROC_NULL 판정).
// 셀 중심 필드: ghost = 2·벽값 − 첫 내부값 (거울).  → 온도 T (Dirichlet)
void fill_wall_mirror(field::Field& f, const parallel::Subdomain& sub, int axis, int side,
                      real_t value);
// 셀 중심 필드: ghost = 벽값.  → 접선 속도 (no-slip 이면 0, lid 면 lid 속도)
void fill_wall_value(field::Field& f, const parallel::Subdomain& sub, int axis, int side,
                     real_t value);
// 셀 중심 필드: ghost = 첫 내부값 (영구배).  → 압력 P, 단열 T, slip 접선 속도
// Neumann 비영 구배: ghost = 내부 − Δ·gradient (minus 면) / 내부 + Δ·gradient (plus 면), Δ = ghost·내부 셀중심 거리
// (등온속 벽 ∂θ/∂n = g — 열 경계층 isoflux 벽, 2026-10-11). gradient 는 "축 양의 방향" 도함수 ∂f/∂x_axis.
void fill_wall_gradient(field::Field& f, const parallel::Subdomain& sub, int axis, int side,
                        real_t gradient, real_t spacing);
void fill_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub, int axis,
                             int side);
// 그 축의 면 필드(axis 0 → U, 1 → V, 2 → W)의 벽 면 자체를 0 으로 (불침투).
// 슬롯 비대칭 (docs/conventions.md §3): minus 벽면 = 내부 슬롯 halo_width,
// plus 벽면 = ghost 슬롯 n_total − halo_width. minus 바깥 면(슬롯 0)은 아무
// 스텐실도 읽지 않는 미사용 슬롯이라 건드리지 않는다.
void zero_wall_faces(field::Field& f, const parallel::Subdomain& sub, int axis, int side);

// ── z 전용 이름 (= axis 2, minus 그리고 plus 순 호출; 기존 앱이 쓴다) ──────────────────────────────
void fill_z_wall_mirror(field::Field& f, const parallel::Subdomain& sub,
                        real_t value_bottom, real_t value_top);
void fill_z_wall_value(field::Field& f, const parallel::Subdomain& sub,
                       real_t value_bottom, real_t value_top);
void fill_z_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub);
void zero_z_wall_faces(field::Field& w, const parallel::Subdomain& sub);

}  // namespace mpmstd::boundary
