#pragma once
// 운동량 방정식 solve — Solver(밴드·스테이지·증분) 공유 + 세 진입점:
//   solve()                   : legacy OB 간이 스킴 (legacy_ob/solve.cpp — rbc 전용, 은퇴 예정)
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
// * 경계 가정: x·y 주기(순환 TDMA), z 벽 — U·V 는 fold_z_wall_value,
//   W 는 상단 fold + 하단 벽면 항등행 (system/system.hpp 의 ⚠ 참조).
// * 수식 출처: MPM-STD_C momentum OB 경로 1:1 (ref_03 §1.5).
//
// 사용 (매 스텝):
//   rhs_U.fill(0); rhs_V.fill(0); rhs_W.fill(0);
//   momentum::add_advection_diffusion_cell(rhs_U, U, U,V,W, grid, sub, nu, dt);
//   momentum::add_advection_diffusion_cell(rhs_V, V, U,V,W, grid, sub, nu, dt);
//   momentum::add_advection_diffusion_z_face(rhs_W, W, U,V,W, grid, sub, nu, dt);
//   pressure::add_gradient_constant_property(rhs_U, rhs_V, rhs_W, P, grid, sub, dt);
//   (+ 케이스 강제항)
//   momentum::solve(solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V, W, nu, dt);
//   (U, V, W 교환 + 벽 처리 → 투영)

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

struct Solver {
  explicit Solver(const parallel::Subdomain& sub);
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  linear_solver::TdmaPlan plan_x, plan_y, plan_z;
  // 밴드·스테이지는 그 빌드의 백엔드 메모리에 산다 (CPU=호스트, GPU=디바이스).
  field::Buffer band_a, band_b, band_c, band_d;  // [sys + row·nsys]
  field::Buffer stage_first, stage_second;       // 스윕 핑퐁 (전체 모양)

  // 증분 δU, δV, δW — 블록 커플링의 스텐실이 이웃 증분을 읽으므로
  // halo 교환이 필요해 Field 로 소유한다 (벽 ghost 는 "증분 = 0" 정책).
  field::Field delta_u, delta_v, delta_w;
};

void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs_u, const field::Field& rhs_v,
           const field::Field& rhs_w, field::Field& U, field::Field& V,
           field::Field& W, real_t viscosity, real_t dt);

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
