#include <cstddef>
#include <stdexcept>
#include <string>

// 구간 타이머 (Solver::timer 가 있을 때만; 없으면 no-op) — plan.md §1.1 의 "한 단계 쪼개기"
#define MOMENTUM_TIMER_BEGIN(name) \
  do { if (solver.timer) solver.timer->begin(name); } while (0)
#define MOMENTUM_TIMER_END(name) \
  do { if (solver.timer) solver.timer->end(name); } while (0)

#include "boundary/wall.hpp"
#include "common/field_checks.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/momentum/solve_kernels.hpp"
#include "equation/momentum/system/system.hpp"
#include "equation/momentum/wall_rows.hpp"
#include "equation/momentum/system/system.hpp"
#include "equation/sweep_kernels.hpp"
#include "equation/band_layout.hpp"
// [최적화 2026-10-18] adi_sweeps_variable_property: D 밴드 핑퐁(scatter_x/y 제거), scatter_z 타일 전치, dW 는 W += δW 융합.

// 가변 점성(NOB) 운동량 solve — 오케스트레이션 전용 TU (CPU/GPU 공용).
//
// OB solve 와의 차이 (참조 solve_momentum_variable_property 1:1):
//   ① **전진 대입** dU → dV → dW (Gauss-Seidel): 직전에 푼 증분을 다음 성분
//      우변에 주입(M21/M31/M32)한 뒤 그 성분의 ADI 를 푼다 — 벽수직 dW 가
//      마지막이라 모든 커플링이 z-음해를 통과한다.
//   ② 사후 상삼각 커플링(M23/M12+M13)은 가변 μ·1/ρ̂ 판 (block_couple_*_variable).
//   ③ rhs_u/v/w 는 **비-const**: 전진 커플링이 주입된다 (매 스텝 재조립 전제).
// 벽 닫기(fold/핀)와 증분 ghost 정책(δ=0)은 OB 와 동일.

namespace mpmstd::equation::momentum {

namespace {

// 한 축 스윕의 풀이: 주기 축 → 순환 TDMA, 벽 축 → fold(δghost=0) + (스윕 축이
// own 축이면 minus 벽면 행 항등) + 비주기 TDMA. 어느 축이 벽인지는 Topology 의
// periodic 플래그가 말한다 (입력 [boundary] → main 의 Topology 생성 인자).
void solve_sweep(const Solver& solver, linear_solver::TdmaPlan& plan, real_t* A, real_t* B,
                 real_t* C, real_t* D, const parallel::Subdomain& sub, const int axis,
                 const int own_axis, const int nsys, const int nrow) {
  if (sub.topology().axis(axis).periodic) {
    linear_solver::solve_cyclic_many(plan, A, B, C, D, nsys, nrow);
    return;
  }
  if (own_axis == axis) {
    // 법선 성분: 닫기 계수 없음 (minus 벽면 행은 항등, plus 쪽 벽면 = ghost 슬롯 δ=0)
    fold_wall_value(A, B, C, sub, axis, nsys, nrow);
    pin_minus_wall_face_row(A, B, C, D, sub, axis, nsys);
  } else {
    fold_wall(A, B, C, sub, axis, nsys, nrow, solver.wall_closure[axis][0],
              solver.wall_closure[axis][1]);
  }
  linear_solver::solve_many(plan, A, B, C, D, nsys, nrow);
}

// 증분 δq 의 벽 ghost (halo 교환 뒤): 법선 성분은 벽면 = 0; 접선 성분은 닫기 계수와
// 짝 — 0: ghost = 0, −1: 거울(ghost = −δq₁), +1: 영구배(ghost = δq₁).
void close_delta_wall_ghosts(const Solver& solver, field::Field& delta,
                             const int component_axis, const parallel::Subdomain& sub) {
  for (int axis = 0; axis < 3; ++axis) {
    if (sub.topology().axis(axis).periodic) continue;
    for (int side = 0; side < 2; ++side) {
      if (axis == component_axis) {
        boundary::zero_wall_faces(delta, sub, axis, side);
        continue;
      }
      const real_t closure = solver.wall_closure[axis][side];
      if (closure < 0) boundary::fill_wall_mirror(delta, sub, axis, side, 0.0);
      else if (closure > 0) boundary::fill_wall_zero_gradient(delta, sub, axis, side);
      else boundary::fill_wall_value(delta, sub, axis, side, 0.0);
    }
  }
}

void adi_sweeps_variable_property(Solver& solver, const grid::Grid& grid,
                         const parallel::Subdomain& sub, const real_t* rhs_data,
                         const field::Field& q, const field::Field& U,
                         const field::Field& V, const field::Field& W,
                         const field::Field& viscosity,
                         const field::Field& inverse_density, const int own_axis,
                         const real_t Cmu, const real_t dt, real_t* delta_out,
                         real_t* accumulate_into) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  real_t* A = solver.workspace->slot(0);   // [최적화 2026-10-23] 공용 Workspace 슬롯 (plan.md §2.3)
  real_t* B = solver.workspace->slot(1);
  real_t* C = solver.workspace->slot(2);
  // D 핑퐁: x → D0, y 는 D0 를 읽어 D1, z 는 D1 을 읽어 D0 (band_layout.hpp).
  real_t* D0 = solver.workspace->slot(3);
  real_t* D1 = solver.workspace->slot(4);
  // 구간 이름 "momentum.dU.build_x" 등 (own_axis 0/1/2 = dU/dV/dW)
  const std::string prefix =
      std::string("momentum.") + (own_axis == 0 ? "dU" : own_axis == 1 ? "dV" : "dW") + ".";
  const auto name = [&](const char* call) { return prefix + call; };

  // x 스윕 (이송 = U) — source = rhs 필드
  MOMENTUM_TIMER_BEGIN(name("build_x").c_str());
  build_x_sweep_variable_property(A, B, C, D0, rhs_data, q.data(), U.data(), viscosity.data(),
                         inverse_density.data(), grid, sub, own_axis, Cmu, dt);
  MOMENTUM_TIMER_END(name("build_x").c_str());
  MOMENTUM_TIMER_BEGIN(name("tdma_x").c_str());
  solve_sweep(solver, solver.plan_x, A, B, C, D0, sub, 0, own_axis, i2 * i3, i1);
  MOMENTUM_TIMER_END(name("tdma_x").c_str());

  // y 스윕 (이송 = V) — source = x 밴드 D0
  MOMENTUM_TIMER_BEGIN(name("build_y").c_str());
  build_y_sweep_variable_property(A, B, C, D1, D0, q.data(), V.data(),
                         viscosity.data(), inverse_density.data(), grid, sub,
                         own_axis, Cmu, dt, SourceLayout::BandX);
  MOMENTUM_TIMER_END(name("build_y").c_str());
  MOMENTUM_TIMER_BEGIN(name("tdma_y").c_str());
  solve_sweep(solver, solver.plan_y, A, B, C, D1, sub, 1, own_axis, i1 * i3, i2);
  MOMENTUM_TIMER_END(name("tdma_y").c_str());

  // z 스윕 (이송 = W) — source = y 밴드 D1. 벽 닫기는 OB 와 동일 (system.hpp ⚠)
  MOMENTUM_TIMER_BEGIN(name("build_z").c_str());
  build_z_sweep_variable_property(A, B, C, D0, D1, q.data(), W.data(),
                         viscosity.data(), inverse_density.data(), grid, sub,
                         own_axis, Cmu, dt, SourceLayout::BandY);
  MOMENTUM_TIMER_END(name("build_z").c_str());
  MOMENTUM_TIMER_BEGIN(name("tdma_z").c_str());
  solve_sweep(solver, solver.plan_z, A, B, C, D0, sub, 2, own_axis, i1 * i2, i3);
  MOMENTUM_TIMER_END(name("tdma_z").c_str());
  // 산포(타일 전치) — accumulate_into 가 있으면 q += δ 도 여기서 (δW: 이후 커플링은 δW 만 읽고 W 는 안 읽는다)
  MOMENTUM_TIMER_BEGIN(name("scatter_z").c_str());
  equation::scatter_z(delta_out, D0, sub, accumulate_into);
  MOMENTUM_TIMER_END(name("scatter_z").c_str());
}

void adi_sweeps_constant(Solver& solver, const grid::Grid& grid,
                         const parallel::Subdomain& sub, const real_t* rhs_data,
                         const field::Field& q, const field::Field& U,
                         const field::Field& V, const field::Field& W,
                         const int own_axis, const real_t viscosity, const real_t dt,
                         real_t* delta_out, real_t* accumulate_into) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  real_t* A = solver.workspace->slot(0);   // [최적화 2026-10-23] 공용 Workspace 슬롯 (plan.md §2.3)
  real_t* B = solver.workspace->slot(1);
  real_t* C = solver.workspace->slot(2);
  // [최적화 2026-10-18] 가변물성 판과 같은 D 밴드 핑퐁(scatter_x/y 없음) + 타일 scatter_z(+ q += δ)
  real_t* D0 = solver.workspace->slot(3);
  real_t* D1 = solver.workspace->slot(4);

  build_x_sweep_constant_property(A, B, C, D0, rhs_data, q.data(), U.data(), grid, sub,
                                  own_axis, viscosity, dt);
  solve_sweep(solver, solver.plan_x, A, B, C, D0, sub, 0, own_axis, i2 * i3, i1);

  build_y_sweep_constant_property(A, B, C, D1, D0, q.data(), V.data(), grid, sub, own_axis,
                                  viscosity, dt, SourceLayout::BandX);
  solve_sweep(solver, solver.plan_y, A, B, C, D1, sub, 1, own_axis, i1 * i3, i2);

  build_z_sweep_constant_property(A, B, C, D0, D1, q.data(), W.data(), grid, sub, own_axis,
                                  viscosity, dt, SourceLayout::BandY);
  solve_sweep(solver, solver.plan_z, A, B, C, D0, sub, 2, own_axis, i1 * i2, i3);
  equation::scatter_z(delta_out, D0, sub, accumulate_into);
}

}  // namespace

std::size_t Solver::required_slot_elements(const parallel::Subdomain& sub) {
  return static_cast<std::size_t>(sub.x().n_local) * sub.y().n_local * sub.z().n_local;
}

// 생성자 (2026-10-23 legacy_ob/solve.cpp 에서 옮김). 밴드는 공용 Workspace 에서 빌린다 (plan.md A 단계).
Solver::Solver(const parallel::Subdomain& sub, Workspace& ws)
    : delta_u(sub.x().n_total, sub.y().n_total, sub.z().n_total, "dU"),
      delta_v(sub.x().n_total, sub.y().n_total, sub.z().n_total, "dV"),
      delta_w(sub.x().n_total, sub.y().n_total, sub.z().n_total, "dW"),
      workspace(&ws) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  plan_x.create(i2 * i3, sub.topology().x().comm);
  plan_y.create(i1 * i3, sub.topology().y().comm);
  plan_z.create(i1 * i2, sub.topology().z().comm);
  ws.require(required_slot_elements(sub), "momentum::Solver");
}

void set_wall_closure(Solver& solver, const boundary::BoundaryCondition& bc) {
  using boundary::BcKind;
  const boundary::Field row[3] = {boundary::Field::U, boundary::Field::V, boundary::Field::W};
  for (int axis = 0; axis < 3; ++axis)
    for (int side = 0; side < 2; ++side) {
      real_t closure = 0;
      if (!bc.is_periodic(axis)) {
        // 접선 성분들은 같은 kind 를 가져야 한 계수로 닫을 수 있다 (성분별 계수는 미지원)
        BcKind kind = BcKind::Periodic;
        for (int c = 0; c < 3; ++c) {
          if (c == axis) continue;
          const BcKind k = bc.at(row[c], axis, side).kind;
          if (kind == BcKind::Periodic) kind = k;
          else if (k != kind)
            throw std::invalid_argument(
                "momentum::set_wall_closure: tangential velocity components on one wall "
                "must share the BC kind (axis " + std::to_string(axis) + ")");
        }
        // Dirichlet(유입 접선 포함) → ghost 정책 짝, Neumann(slip) → +1,
        // Outflow(유출: 값이 명시 갱신되는 Dirichlet) → 0 (δ_ghost = 0)
        if (kind == BcKind::Neumann) closure = 1;
        else if (kind == BcKind::Dirichlet)
          closure = bc.velocity_ghost == boundary::VelocityGhost::Mirror ? real_t(-1) : 0;
        else closure = 0;
      }
      solver.wall_closure[axis][side] = closure;
    }
}

void solve_constant_property(Solver& solver, const grid::Grid& grid,
                             const parallel::Subdomain& sub, field::Field& rhs_u,
                             field::Field& rhs_v, field::Field& rhs_w,
                             field::Field& U, field::Field& V, field::Field& W,
                             const real_t viscosity, const real_t dt) {
  check_field_matches("momentum::solve_constant_property", rhs_u, sub);
  check_field_matches("momentum::solve_constant_property", rhs_v, sub);
  check_field_matches("momentum::solve_constant_property", rhs_w, sub);
  if (solver.delta_u.size() != rhs_u.size())
    throw std::invalid_argument(
        "momentum::solve_constant_property: Solver was created for a different "
        "subdomain");
  solver.workspace->require(Solver::required_slot_elements(sub), "momentum::solve");

  // 전진 대입 dU → dV → dW (solve_variable_property 1:1 — 물성 자리만 상수)
  adi_sweeps_constant(solver, grid, sub, rhs_u.data(), U, U, V, W, /*own_axis=*/0,
                      viscosity, dt, solver.delta_u.data(), nullptr);
  sub.exchange_halo(solver.delta_u.data());
  close_delta_wall_ghosts(solver, solver.delta_u, 0, sub);

  couple_rhs_v_from_delta_u_constant_property(rhs_v, solver.delta_u, V, grid, sub,
                                              viscosity, dt);
  adi_sweeps_constant(solver, grid, sub, rhs_v.data(), V, U, V, W, /*own_axis=*/1,
                      viscosity, dt, solver.delta_v.data(), nullptr);
  sub.exchange_halo(solver.delta_v.data());
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);

  couple_rhs_w_from_delta_uv_constant_property(rhs_w, solver.delta_u, solver.delta_v,
                                               W, grid, sub, viscosity, dt);
  adi_sweeps_constant(solver, grid, sub, rhs_w.data(), W, U, V, W, /*own_axis=*/2,
                      viscosity, dt, solver.delta_w.data(), W.data());   // W += δW 를 scatter_z 에서
  sub.exchange_halo(solver.delta_w.data());
  close_delta_wall_ghosts(solver, solver.delta_w, 2, sub);

  block_couple_delta_v_constant_property(solver.delta_v, solver.delta_w, V, grid, sub,
                                         viscosity, dt);
  sub.exchange_halo(solver.delta_v.data());
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);
  block_couple_delta_u_constant_property(solver.delta_u, solver.delta_v,
                                         solver.delta_w, U, grid, sub, viscosity, dt);

  equation::add_increment(U, solver.delta_u.data(), sub);
  equation::add_increment(V, solver.delta_v.data(), sub);
  // W += δW 는 dW 의 scatter_z 에서 이미
}

void solve_variable_property(Solver& solver, const grid::Grid& grid,
                    const parallel::Subdomain& sub, field::Field& rhs_u,
                    field::Field& rhs_v, field::Field& rhs_w, field::Field& U,
                    field::Field& V, field::Field& W, const field::Field& viscosity,
                    const field::Field& inverse_density, const real_t Cmu,
                    const real_t dt) {
  check_field_matches("momentum::solve_variable_property", rhs_u, sub);
  check_field_matches("momentum::solve_variable_property", rhs_v, sub);
  check_field_matches("momentum::solve_variable_property", rhs_w, sub);
  check_field_matches("momentum::solve_variable_property", viscosity, sub);
  check_field_matches("momentum::solve_variable_property", inverse_density, sub);
  if (solver.delta_u.size() != rhs_u.size())
    throw std::invalid_argument(
        "momentum::solve_variable_property: Solver was created for a different subdomain");

  // ── 전진 대입 dU → dV → dW ──
  // ① dU (own x) — 커플링 없음
  adi_sweeps_variable_property(solver, grid, sub, rhs_u.data(), U, U, V, W, viscosity,
                      inverse_density, /*own_axis=*/0, Cmu, dt,
                      solver.delta_u.data(), nullptr);
  MOMENTUM_TIMER_BEGIN("momentum.dU.halo");
  sub.exchange_halo(solver.delta_u.data());
  MOMENTUM_TIMER_END("momentum.dU.halo");
  MOMENTUM_TIMER_BEGIN("momentum.dU.wall_ghost");
  close_delta_wall_ghosts(solver, solver.delta_u, 0, sub);
  MOMENTUM_TIMER_END("momentum.dU.wall_ghost");

  // ② dV (own y) — M21(δU) 주입 후 풀이
  MOMENTUM_TIMER_BEGIN("momentum.dV.couple_rhs");
  couple_rhs_v_from_delta_u_variable_property(rhs_v, solver.delta_u, V, viscosity,
                                     inverse_density, grid, sub, Cmu, dt);
  MOMENTUM_TIMER_END("momentum.dV.couple_rhs");
  adi_sweeps_variable_property(solver, grid, sub, rhs_v.data(), V, U, V, W, viscosity,
                      inverse_density, /*own_axis=*/1, Cmu, dt,
                      solver.delta_v.data(), nullptr);
  MOMENTUM_TIMER_BEGIN("momentum.dV.halo");
  sub.exchange_halo(solver.delta_v.data());
  MOMENTUM_TIMER_END("momentum.dV.halo");
  MOMENTUM_TIMER_BEGIN("momentum.dV.wall_ghost");
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);
  MOMENTUM_TIMER_END("momentum.dV.wall_ghost");

  // ③ dW (own z, 벽) — M31(δU) + M32(δV) 주입 후 풀이
  MOMENTUM_TIMER_BEGIN("momentum.dW.couple_rhs");
  couple_rhs_w_from_delta_uv_variable_property(rhs_w, solver.delta_u, solver.delta_v, W,
                                      viscosity, inverse_density, grid, sub, Cmu, dt);
  MOMENTUM_TIMER_END("momentum.dW.couple_rhs");
  adi_sweeps_variable_property(solver, grid, sub, rhs_w.data(), W, U, V, W, viscosity,
                      inverse_density, /*own_axis=*/2, Cmu, dt,
                      solver.delta_w.data(), W.data());   // W += δW 를 scatter_z 에서
  MOMENTUM_TIMER_BEGIN("momentum.dW.halo");
  sub.exchange_halo(solver.delta_w.data());
  MOMENTUM_TIMER_END("momentum.dW.halo");
  // 하단 벽면(내부 행 k=h)의 가짜 증분 제거 포함 — 커플링이 소비하기 전
  // (벽 전단 1/dmz 증폭으로 U 스파이크가 되는 경로; 참조 Report/27)
  MOMENTUM_TIMER_BEGIN("momentum.dW.wall_ghost");
  close_delta_wall_ghosts(solver, solver.delta_w, 2, sub);
  MOMENTUM_TIMER_END("momentum.dW.wall_ghost");

  // ── 사후 상삼각: δV ← δW, 동기화, δU ← (보정된 δV) + δW. δW 는 무보정 ──
  MOMENTUM_TIMER_BEGIN("momentum.post.block_couple_v");
  block_couple_delta_v_variable_property(solver.delta_v, solver.delta_w, V, viscosity,
                                inverse_density, grid, sub, Cmu, dt);
  MOMENTUM_TIMER_END("momentum.post.block_couple_v");
  MOMENTUM_TIMER_BEGIN("momentum.post.halo_dV");
  sub.exchange_halo(solver.delta_v.data());
  MOMENTUM_TIMER_END("momentum.post.halo_dV");
  MOMENTUM_TIMER_BEGIN("momentum.post.wall_ghost_dV");
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);
  MOMENTUM_TIMER_END("momentum.post.wall_ghost_dV");
  MOMENTUM_TIMER_BEGIN("momentum.post.block_couple_u");
  block_couple_delta_u_variable_property(solver.delta_u, solver.delta_v, solver.delta_w, U,
                                viscosity, inverse_density, grid, sub, Cmu, dt);
  MOMENTUM_TIMER_END("momentum.post.block_couple_u");

  // 속도 갱신 (내부만) — ghost 재동기화는 호출자의 다음 sync 몫
  MOMENTUM_TIMER_BEGIN("momentum.post.add_increment");
  equation::add_increment(U, solver.delta_u.data(), sub);
  equation::add_increment(V, solver.delta_v.data(), sub);
  // W += δW 는 dW 의 scatter_z 에서 이미 (adi_sweeps_variable_property accumulate_into)
  MOMENTUM_TIMER_END("momentum.post.add_increment");
}

}  // namespace mpmstd::equation::momentum
