#pragma once
// 전역 벽의 ghost 채움 — 경계항 처리의 "명시(explicit) 반쪽".
//
// ── 설계 규칙 ────────────────────────────────────────────────────────────────
// * 정책 = 함수 이름이다. BcKind/GhostPolicy 같은 enum 도, 입력 파일 키도 없다.
//   어떤 정책을 쓸지는 각 앱의 main 이 함수를 골라 부르는 것으로 표현된다.
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
// ── 왜 z 전용인가 / x·y 벽 확장 경로 ────────────────────────────────────────
// 현재 모든 앱(RBC·DHVC·channel)의 벽은 z 뿐이다. cavity 처럼 x/y 벽이 필요한
// 앱이 생기면 **같은 모양의 함수를 옆에 추가**한다 (fill_x_wall_mirror, ...) —
// D13(축 매개변수화 금지)이라 기존 코드는 건드리지 않는다. 아래 계층은 이미
// 축 대칭이다 (Topology 의 periodic_x/y/z, halo 교환의 MPI_PROC_NULL no-op).
// 단, 진짜 본체는 압력 솔버다: x/y 가 벽이면 그 방향 FFT 를 DCT/DST 로 바꿔야
// 한다 (M8; CaNS initsolver 의 BC 조합별 고유값 표 참조).

#include "common/types.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::boundary {

// 셀 중심 필드: ghost = 2·벽값 − 첫 내부값 (거울).  → 온도 T
void fill_z_wall_mirror(field::Field& f, const parallel::Subdomain& sub,
                        real_t value_bottom, real_t value_top);

// 셀 중심 필드: ghost = 벽값.  → 접선 속도 U, V (no-slip 이면 0)
void fill_z_wall_value(field::Field& f, const parallel::Subdomain& sub,
                       real_t value_bottom, real_t value_top);

// 셀 중심 필드: ghost = 첫 내부값 (영구배).  → 압력 P
void fill_z_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub);

// z-면 필드(W)의 벽 면 자체를 0 으로 (불침투). 슬롯 비대칭에 주의
// (docs/conventions.md §3): 하단 벽면 = 내부 슬롯 k = halo_width,
// 상단 벽면 = ghost 슬롯 k = nz_total − halo_width. 하단 바깥 면(k=0)은
// 아무 스텐실도 읽지 않는 미사용 슬롯이라 건드리지 않는다.
void zero_z_wall_faces(field::Field& w, const parallel::Subdomain& sub);

}  // namespace mpmstd::boundary
