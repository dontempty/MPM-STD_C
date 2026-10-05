#pragma once
// BoundaryCondition 테이블 → ghost 채움 함수 선택 (wall.hpp 의 "정책 = 함수 이름"
// 위에 얹힌 한 층). 비주기 축마다 필드 종류별로:
//
//   속도 — 축 a 의 법선 성분(a=0→U, 1→V, 2→W): 벽 면 자체 = 0 (zero_wall_faces;
//          불투과. Dirichlet 값은 0 만 허용 — 분사/흡입은 미지원이라 throw)
//          접선 성분: Dirichlet → ghost = 벽값 (fill_wall_value, no-slip/lid)
//                     Neumann  → ghost = 첫 내부값 (fill_wall_zero_gradient, slip)
//   스칼라 T — Dirichlet → mirror(2v − f₁),  Neumann → zero_gradient (단열)
//   압력 P  — 항상 zero_gradient (테이블은 Neumann 으로 검증돼 있다)
//
// 호출 순서 규칙(wall.hpp): **halo 교환 먼저, 이 함수 나중.** 주기 축은 건너뛴다.
// CPU/GPU 공용 (wall.* 쌍둥이를 부를 뿐).

#include "boundary/boundary_condition.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::boundary {

void apply_velocity_wall_ghosts(field::Field& U, field::Field& V, field::Field& W,
                                const BoundaryCondition& bc,
                                const parallel::Subdomain& sub);

// 셀 중심 스칼라 (온도 등). which 는 테이블의 어느 필드 행을 볼지 (보통 Field::T).
void apply_scalar_wall_ghosts(field::Field& f, Field which, const BoundaryCondition& bc,
                              const parallel::Subdomain& sub);

void apply_pressure_wall_ghosts(field::Field& P, const BoundaryCondition& bc,
                                const parallel::Subdomain& sub);

}  // namespace mpmstd::boundary
