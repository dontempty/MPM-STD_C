#pragma once
// 열린 경계(유입·유출·자유류 상단) — 경계층 앱의 "앱이 매 스텝 공급하는" BC.
// (2026-10-06, docs/Design/20261006_boundary_layer_validation.md §3)
//
// 축 관례 고정: x− 유입, x+ 유출, z+ 자유류 (BoundaryCondition 의 kind 가 Inflow/Outflow/
// Continuity 인 면). wall.hpp 와 같은 규칙 — 그 면을 소유한 랭크에서만 동작(이웃 == MPI_PROC_NULL),
// halo 교환 뒤에 부른다. CPU(open.cpp)/GPU(open.cu) 쌍둥이.
//
//   set_inflow               U(x− 면) = u(z), V·W ghost = 프로파일(mirror 또는 value)
//   update_outflow_convective  ∂φ/∂t + U_c ∂φ/∂x = 0 — 유출면(U) / ghost(V, W) 를 uⁿ 으로 명시 갱신
//   set_top_normal_from_continuity  W(z+ 면) = W(아래 면) − dz·(∂U/∂x + ∂V/∂y)  (상단 셀 발산 0; ZPG 엔 부적합)
//   set_top_normal_profile   W(z+ 면) = w_top(x) — 앱이 주는 자유류 프로파일 (Blasius w∞ / U∞ dδ*/dx)
//   correct_outflow_flux     ∮u·n = 0 강제: U(x+ 면) += (Q_in − Q_top − Q_out)/A_out  (Neumann Poisson 양립)
//                            반환 = 더한 보정량 (모니터용)

#include <vector>

#include "boundary/boundary_condition.hpp"
#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::boundary {

// 유입 프로파일 — 로컬 z 인덱스(ghost 포함, 크기 nz_total) 로 색인한 호스트 배열.
struct InflowProfile {
  std::vector<real_t> u, v, w;
};

void set_inflow(field::Field& U, field::Field& V, field::Field& W, const InflowProfile& profile,
                VelocityGhost ghost, const parallel::Subdomain& sub);

void update_outflow_convective(field::Field& U, field::Field& V, field::Field& W,
                               real_t convective_velocity, real_t dt, const grid::Grid& grid,
                               const parallel::Subdomain& sub);

void set_top_normal_from_continuity(field::Field& W, const field::Field& U,
                                    const field::Field& V, const grid::Grid& grid,
                                    const parallel::Subdomain& sub);

// z+ 면 W = w_top[i] (로컬 x 인덱스, ghost 포함, 크기 nx_total) — FreeStream kind 용
void set_top_normal_profile(field::Field& W, const std::vector<real_t>& w_top,
                            const parallel::Subdomain& sub);

real_t correct_outflow_flux(field::Field& U, const field::Field& W, const grid::Grid& grid,
                            const parallel::Subdomain& sub);

}  // namespace mpmstd::boundary
