#pragma once
// 운동량 방정식 solve — Solver(TDMA 플랜·공용 Workspace) + 두 진입점:
//   (legacy OB 간이 스킴 solve() 와 apps/rbc 드라이버는 2026-10-23 삭제 — 쓰는 앱이 없었다)
//   solve_variable_property() : 충실 가변물성 (solve.cpp)
//   solve_constant_property() : 충실 상수물성, 물성 배열 없음 (solve.cpp)
//
//   각 성분 q ∈ {U, V, W}:
//     (I − ½dtνLx − ¼dtCx)(I − ½dtνLy − ¼dtCy)(I − ½dtνLz − ¼dtCz) δq = rhs_q
//   그 뒤 블록 커플링(대류 교차항; MPM-STD_C block_couple 1:1):
//     δV −= dt·¼·(δW·∂V/∂z)        (M23)
//     δU −= dt·¼·(δV·∂U/∂y + δW·∂U/∂z)   (M12 + M13 — 보정된 δV 사용)
//   마지막으로 U += δU, V += δV, W += δW (내부만; ghost 재동기화는 호출자 몫).
//
// ★ 이 증분 교차 대류가 streak↔vortex 커플링이다 — 빼면 채널 난류가
//   재층류화한다 (MPM-STD_C kernels.hpp:133-135; 블록 결합 부재가 과거
//   재층류화의 원인이었음). 벽수직 δW 는 사후 보정을 받지 않는다.
// * rhs_u/v/w 는 호출자가 조립해 const 로 (운동량 명시항 → 압력 구배
//   pressure::add_gradient_constant_property → 채널 강제 physics::channel_forcing 순).
// * 경계: 축마다 Topology 의 periodic 플래그로 고른다 (2026-10-05 BC 프레임워크) —
//   주기 축 = 순환 TDMA, 벽 축 = 접선 성분 fold(δghost=0) / 법선 성분 fold +
//   minus 벽면 항등행 (system/system.hpp 의 ⚠ 참조). 기존 앱은 x·y 주기, z 벽.
// * 수식 출처: MPM-STD_C momentum OB 경로 1:1 (ref_03 §1.5).
//
// 사용 (매 스텝, 상수물성 예 — apps/dhvc):
//   momentum::build_constant_property_rhs(rhs_U, rhs_V, rhs_W, U, V, W, P, grid, sub, nu, dt, &T);
//   momentum::solve_constant_property(solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V, W, nu, dt);
//   (U, V, W 교환 + 벽 처리 → 투영)

#include "common/types.hpp"
#include "boundary/boundary_condition.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "equation/workspace.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "common/phase_timer.hpp"

namespace mpmstd::equation::momentum {

struct Solver {
  Solver(const parallel::Subdomain& sub, Workspace& workspace);
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;
  // 이 솔버가 요구하는 Workspace 슬롯 크기 (밴드 [sys + row·nsys] = i1·i2·i3) — 앱이 최대값으로 Workspace 를 만든다.
  static std::size_t required_slot_elements(const parallel::Subdomain& sub);

  linear_solver::TdmaPlan plan_x, plan_y, plan_z;
  // [최적화 2026-10-23] 밴드 A·B·C·D0·D1(핑퐁)은 솔버가 소유하지 않고 공용 Workspace 슬롯 0~4 를 빌려 쓴다
  //   (docs/Memory/plan.md). 성분 dU·dV·dW 가 같은 슬롯을 차례로 쓴다 — 밴드는 한 스윕 안에서만 산다.
  Workspace* workspace = nullptr;

  // [메모리 2026-10-26] 증분 δU δV δW 는 솔버가 소유하지 않는다 — 호출자가 넘긴 rhs_u/v/w 의 메모리를
  //   그대로 쓴다 (들어올 때 rhs, 나갈 때 δ). 각 성분의 x 스윕 조립이 rhs 를 밴드로 읽고 나면 rhs 는 죽고,
  //   δ 는 z 스윕 뒤 scatter_z 가 처음 쓴다. 그래서 solve_* 는 rhs 를 덮어쓴다 — 호출자는 solve 뒤 rhs 값을
  //   기대하면 안 된다 (앱은 매 스텝 build_*_rhs 로 새로 만든다). docs/Memory/plan.md §9.

  // 접선 성분 벽 행 음해 닫기 계수 [axis][side] (wall_rows.hpp fold_wall):
  //   0 = δghost 0 (value ghost 와 짝; 기본 — 기존 앱 비트 불변)
  //  −1 = 거울 ghost 일치 ([boundary] velocity_ghost = mirror)
  //  +1 = 영구배 ghost 일치 (slip)
  // 법선 성분은 closure 와 무관하게 minus 벽면 행 항등(pin) + plus 쪽 C = 0.
  real_t wall_closure[3][2] = {{0, 0}, {0, 0}, {0, 0}};

  // 확장성 측정용 (docs/Scalability/plan.md §1.1): nullptr 이 아니면 solve_variable_property 가
  // 자기 안의 호출(빌드·TDMA·scatter ×3축 ×3성분, halo, 벽 ghost, 커플링, 증분 적용)을
  // "momentum.<성분>.<호출>" 구간으로 잰다. 수치는 불변 — 경계마다 동기화만 추가된다.
  mpmstd::common::PhaseTimer* timer = nullptr;
};

// BoundaryCondition 테이블에서 wall_closure 를 채운다: 비주기 축의 접선 Dirichlet →
// (velocity_ghost == Mirror ? −1 : 0), Neumann(slip) → +1. 주기 축 0.
void set_wall_closure(Solver& solver, const boundary::BoundaryCondition& bc);

// ── 가변 점성(NOB) 경로 — 같은 Solver 를 재사용한다 ─────────────────────────
// OB 와 달리 ① 전진 대입 dU→dV→dW (M21/M31/M32 를 직전 증분으로 우변에 주입 —
// 그래서 rhs_* 가 비-const 다; 매 스텝 재조립 전제) ② 사후 커플링은 가변
// μ̂·1/ρ̂ 판. 상수물성 앱(OB·DHVC)은 이 함수와 물성 필드를 쓰지 않는다.
// Cmu = √(Pr/Ra). 사용 순서는 solve_variable_property.cpp 머리말 참조.
// ── 상수물성 충실 경로 (plan §18-1) — solve_variable_property 과 같은 구조(전진 대입 +
// 사후 블록 커플링, 참조 1:1)를 물성 배열 없이 돈다. viscosity = 스칼라 ν.
// "가변 + μ̂≡1 배열" 과 비트 일치가 검증 게이트. DHVC·channel 이 쓴다.
void solve_constant_property(Solver& solver, const grid::Grid& grid,
                             const parallel::Subdomain& sub, field::Field& rhs_u,
                             field::Field& rhs_v, field::Field& rhs_w,
                             field::Field& U, field::Field& V, field::Field& W,
                             real_t viscosity, real_t dt);

void solve_variable_property(Solver& solver, const grid::Grid& grid,
                    const parallel::Subdomain& sub, field::Field& rhs_u,
                    field::Field& rhs_v, field::Field& rhs_w, field::Field& U,
                    field::Field& V, field::Field& W, const field::Field& viscosity,
                    const field::Field& inverse_density, real_t Cmu, real_t dt);

}  // namespace mpmstd::equation::momentum
