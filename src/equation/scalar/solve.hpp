#pragma once
// 에너지(온도) 방정식의 solve — 델타형 3-스윕 ADI.
//
//   (I − ½dtαLx)(I − ½dtαLy)(I − ½dtαLz) δθ = rhs,     θⁿ⁺¹ = θⁿ + δθ
//
// * rhs 는 호출자(main/test)가 rhs/rhs.hpp 의 += 함수들로 조립해 **const** 로
//   넘긴다 (계획 §5.5 — solve 는 RHS 를 바꾸지 않는다).
// * 확산 CN + 대류 명시. 인수분해 교차항은 O(dt³)/스텝이라 전체 2차 유지.
// * 경계 가정은 현재 전 앱(RBC·DHVC·channel) 공통: **x·y 주기** → 순환 TDMA,
//   **z 벽** → fold_z_wall_mirror (T 의 ghost 정책과 쌍 — conventions.md §6).
//   x/y 벽이 필요한 앱(cavity)이 생기면 M5 와 같은 방식으로 fold_x_wall_* 를
//   system/ 에 추가하고 여기 스윕 블록에서 바꿔 끼운다.
// * TDMA 는 중간계층(linear_solver/tdma/tdma.hpp)만 본다 — 백엔드 교체와 무관.
// * 수식 출처: MPM-STD_C OB 경로 1:1 (Report/study/ref_03 §2.6).
//
// 사용 (매 스텝):
//   sub.exchange_halo(T.data());
//   boundary::fill_z_wall_mirror(T, sub, v_bottom, v_top);
//   rhs.fill(0);
//   equation::scalar::add_diffusion(rhs, T, grid, sub, alpha, dt);
//   equation::scalar::add_convection(rhs, T, U, V, W, grid, sub, dt);
//   equation::scalar::solve(solver, grid, sub, rhs, T, alpha, dt);

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "boundary/boundary_condition.hpp"
#include "physics/properties.hpp"

namespace mpmstd::equation::scalar {

// TDMA 플랜(통신 상대)과 작업 버퍼 — 한 번 만들어 매 스텝 재사용한다.
struct Solver {
  explicit Solver(const parallel::Subdomain& sub);
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  linear_solver::TdmaPlan plan_x, plan_y, plan_z;
  // 벽 행 음해 닫기 계수 [axis][side] — 0 = drop(PaScaL 충실, 기본: 기존 앱 비트 불변),
  // −1 = Dirichlet 거울 ghost 일치, +1 = Neumann 영구배 ghost 일치. 의미·안정성은
  // scalar_stencil.hpp 의 WallMask::closure 주석. 새 앱은 set_wall_closure 로 ghost
  // 정책과 맞출 것 (cavity: 드롭이면 dt·κ/dx² ≳ 3 에서 폭주 — 실측).
  real_t wall_closure[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  // 밴드·스테이지는 그 빌드의 백엔드 메모리에 산다 (CPU=호스트, GPU=디바이스).
  field::Buffer band_a, band_b, band_c, band_d;  // [sys + row·nsys]
  field::Buffer stage_first, stage_second;       // 전체 모양 스크래치 (δ 핑퐁)
};

// BoundaryCondition 테이블의 필드 which(보통 T)에 맞춰 wall_closure 를 채운다
// (Dirichlet → −1, Neumann → +1, 주기 → 0).
void set_wall_closure(Solver& solver, const boundary::BoundaryCondition& bc,
                      boundary::Field which);

// θⁿ⁺¹ = θⁿ + δθ. T 의 내부만 갱신한다 — ghost 는 다음 스텝의 교환+벽 채움 몫.
void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs, field::Field& T, real_t alpha, real_t dt);

// ── 가변-κ̂ (NOB) 경로 — 같은 Solver 를 재사용한다 ──────────────────────────
// OB 와 달리 rhs 인자가 없다: 첫 스윕이 우변(R − ΣM·θⁿ)을 융합 조립한다.
// 대류도 절반이 밴드(준음해)로 들어간다 — 상수물성 극한(물성 전부 ≡1)에서
// 확산은 OB 와 같은 식이 되지만 대류 이산화는 다르다 (참조 스킴의 성질).
// 상수물성 앱(OB·DHVC)은 이 함수와 PropertyFields 를 쓰지 않는다.
//   사용: update_properties(...) → solve_variable_property(...) → T 교환 + 벽 채움.
// ── 상수물성 충실 경로 (plan §18-1) — solve_variable_property 과 같은 융합 구조를
// 물성 배열 없이 돈다 (κ̂·1/(ρ̂Ĉp)≡1, 미분장≡0). Ct = 스칼라 확산 계수.
void solve_constant_property(Solver& solver, const grid::Grid& grid,
                             const parallel::Subdomain& sub, field::Field& T,
                             const field::Field& U, const field::Field& V,
                             const field::Field& W, real_t Ct, real_t dt);

void solve_variable_property(Solver& solver, const grid::Grid& grid,
                    const parallel::Subdomain& sub, field::Field& T,
                    const field::Field& U, const field::Field& V,
                    const field::Field& W, const physics::PropertyFields& properties,
                    real_t Ct, real_t dt);

}  // namespace mpmstd::equation::scalar
