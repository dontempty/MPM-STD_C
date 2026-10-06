#pragma once
// 열린 경계(유입·유출·자유류 상단) — 경계층 앱의 "앱이 매 스텝 공급하는" BC.
// (2026-10-06, docs/Design/20261006_boundary_layer_validation.md §3)
//
// 축 관례 고정: x− 유입, x+ 유출, z+ 자유류 (BoundaryCondition 의 kind 가 Inflow/Outflow/
// Continuity 인 면). wall.hpp 와 같은 규칙 — 그 면을 소유한 랭크에서만 동작(이웃 == MPI_PROC_NULL),
// halo 교환 뒤에 부른다. CPU(open.cpp)/GPU(open.cu) 쌍둥이.
//
//   set_inflow               U(x− 면) = u(y,z), V·W ghost = 평면값(mirror 또는 value)
//   extract_x_plane          x 평면 하나를 호스트로 (재순환 유입의 재료)
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

// 유입 평면 — 로컬 (y, z) 인덱스(ghost 포함, 크기 ny_total × nz_total, [j·nz_total + k])
// 의 호스트 배열. z 전용 프로파일(Blasius)은 from_profile 로 y 에 복제한다.
struct InflowPlane {
  int ny_total = 0, nz_total = 0;
  std::vector<real_t> u, v, w;
  static InflowPlane from_profile(const std::vector<real_t>& u_z, const std::vector<real_t>& v_z,
                                  const std::vector<real_t>& w_z, int ny_total);
  real_t& at(std::vector<real_t>& f, int j, int k) const { return f[static_cast<std::size_t>(j) * nz_total + k]; }
};

void set_inflow(field::Field& U, field::Field& V, field::Field& W, const InflowPlane& plane,
                VelocityGhost ghost, const parallel::Subdomain& sub);

// x 평면(로컬 i 슬롯)을 호스트 배열 [j·nz_total + k] 로 복사 (재순환 유입 재료; GPU 는 커널+memcpy)
void extract_x_plane(const field::Field& F, int i, std::vector<real_t>& out,
                     const parallel::Subdomain& sub);

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
