#pragma once
// Boussinesq 부력 (OB) — RBC 의 중력축은 z 다 (케이스 정의; DHVC 의 x 중력은
// 그때 add_buoyancy_x 를 옆에 추가한다 — D13, 벽 함수들과 같은 확장 경로).
//
//   rhs_w += dt · T_face,   T_face = 0.5·(dz[k−1]·T(i,j,k) + dz[k]·T(i,j,k−1))/dmz[k]
//
// T 를 W 의 z-면 위치로 **체적가중 보간**하는 것이 핵심이다 — 셀 중심값을 그대로
// 쓰면 θ_c 갭 −46% 를 만든 실측 버그가 된다 (MPM-STD_C 부력 셀중심 버그, E18).
// 자유낙하 무차원이라 계수 1, 기준온도 0 (PaScaL_TCS f90:348 1:1).
// 호출 순서: 운동량 rhs 조립 뒤(+=), momentum::solve 전. T 는 그 스텝에서 이미
// 갱신된 Tⁿ⁺¹ (에너지 먼저 — 참조 루프 순서).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics {

void add_buoyancy_z_constant_property(field::Field& rhs_w, const field::Field& T, const grid::Grid& grid,
                    const parallel::Subdomain& sub, real_t dt);

// ── 가변 밀도(NOB) 판 — PaScaL_TCS mom.f90:711/731 1:1 ──────────────────────
//   rhs_w += dt · (1/ρ̂)_면 · (T_면 + (a₂/a₁)·T_면²·ΔT)
// T 와 1/ρ̂ 둘 다 같은 z-면 체적가중 보간 (셀중심 함정 동일). 2차항은 ρ(T) 의
// 비선형 몫 — density_quadratic_ratio() (글리세롤 a₂≠0; ΔT=0 이면 OB 로 환원).
void add_buoyancy_z_variable_property(field::Field& rhs_w, const field::Field& T,
                             const field::Field& inverse_density,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             real_t density_quadratic_ratio, real_t delta_temperature,
                             real_t dt);

}  // namespace mpmstd::physics
