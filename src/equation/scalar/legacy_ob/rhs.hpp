#pragma once
// ⚠ legacy OB 경로 — 간이 스킴(보존형 RHS + 짝 없는 이류형 음해항). 블록
// 커플링의 평균이 0 이 아니어서 운동량이 dt 에 비례해 샌다(2026-10-04 수지
// 계측). **새 코드에서 쓰지 말 것** — 충실 경로(상위 디렉터리)의
// *_variable_property / *_constant_property 를 쓴다. rbc(OB) 앱 하위호환으로만
// 남아 있고, rbc 전환(Nu 회귀 재검증) 후 디렉터리째 삭제 예정.
// 에너지(온도) 방정식의 명시(explicit) RHS 항 — 전부 += 라서 호출 순서가 자유다
// (계획 §5.5: 덮어쓰기(=) 공개 함수 금지). rhs 는 호출자가 선언하고 매 스텝
// rhs.fill(0) 으로 비운 뒤 여기 함수들로 쌓는다.
//
// 푸는 식 (자유낙하 무차원, α = 1/√(Ra·Pr)):
//   ∂θ/∂t = −∇·(u θ) + α ∇²θ
// 시간 적분은 solve.hpp 의 델타형 ADI (확산 CN, 대류 명시). 그래서 여기의
// 확산 항은 ½ 이 아니라 **라플라시안 전체 × dt** 다 — CN 을 델타형으로 쓰면
//   (I − ½dtαL) δθ = dt·(αLθⁿ − ∇·(uθⁿ))
// 가 되기 때문이다. 수식은 MPM-STD_C OB 경로 1:1 (Report/study/ref_03 §2.6).
//
// 호출 전 T 의 ghost 가 최신이어야 한다: sub.exchange_halo(T.data()) →
// boundary::fill_z_wall_mirror(T, ...) (교환 먼저, 벽 나중). 내부 셀만 쓴다.

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::scalar {

// rhs += dt·α·∇²T.   L_d φ_i = ((φ_{i+1}−φ_i)/dmx_{i+1} − (φ_i−φ_{i−1})/dmx_i)/dx_i
// (dx = cell_width, dmx = center_distance — 비균일 격자 일반형)
void add_diffusion(field::Field& rhs, const field::Field& T, const grid::Grid& grid,
                   const parallel::Subdomain& sub, real_t alpha, real_t dt);

// rhs += −dt·∇·(u T).  보존형 면 플럭스: MAC 면 속도 × 셀값 2점 평균.
// 셀 (i,j,k) 의 −x 면 속도가 U(i,j,k), +x 면이 U(i+1,j,k) (conventions.md §3).
void add_convection(field::Field& rhs, const field::Field& T, const field::Field& U,
                    const field::Field& V, const field::Field& W,
                    const grid::Grid& grid, const parallel::Subdomain& sub, real_t dt);

}  // namespace mpmstd::equation::scalar
