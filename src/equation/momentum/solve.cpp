#include <cstddef>
#include <stdexcept>
#include <string>

#include "boundary/wall.hpp"
#include "common/field_checks.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/momentum/solve_kernels.hpp"
#include "equation/momentum/system/system.hpp"
#include "equation/momentum/wall_rows.hpp"
#include "equation/momentum/system/system.hpp"
#include "equation/sweep_kernels.hpp"

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
                         const real_t Cmu, const real_t dt, real_t* delta_out) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();

  // x 스윕 (이송 = U)
  build_x_sweep_variable_property(A, B, C, D, rhs_data, q.data(), U.data(), viscosity.data(),
                         inverse_density.data(), grid, sub, own_axis, Cmu, dt);
  solve_sweep(solver, solver.plan_x, A, B, C, D, sub, 0, own_axis, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  // y 스윕 (이송 = V)
  build_y_sweep_variable_property(A, B, C, D, solver.stage_first.data(), q.data(), V.data(),
                         viscosity.data(), inverse_density.data(), grid, sub,
                         own_axis, Cmu, dt);
  solve_sweep(solver, solver.plan_y, A, B, C, D, sub, 1, own_axis, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  // z 스윕 (이송 = W) — 벽 닫기는 OB 와 동일 (system.hpp ⚠)
  build_z_sweep_variable_property(A, B, C, D, solver.stage_second.data(), q.data(), W.data(),
                         viscosity.data(), inverse_density.data(), grid, sub,
                         own_axis, Cmu, dt);
  solve_sweep(solver, solver.plan_z, A, B, C, D, sub, 2, own_axis, i1 * i2, i3);
  equation::scatter_z(delta_out, D, sub);
}

void adi_sweeps_constant(Solver& solver, const grid::Grid& grid,
                         const parallel::Subdomain& sub, const real_t* rhs_data,
                         const field::Field& q, const field::Field& U,
                         const field::Field& V, const field::Field& W,
                         const int own_axis, const real_t viscosity, const real_t dt,
                         real_t* delta_out) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();

  build_x_sweep_constant_property(A, B, C, D, rhs_data, q.data(), U.data(), grid, sub,
                                  own_axis, viscosity, dt);
  solve_sweep(solver, solver.plan_x, A, B, C, D, sub, 0, own_axis, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  build_y_sweep_constant_property(A, B, C, D, solver.stage_first.data(), q.data(),
                                  V.data(), grid, sub, own_axis, viscosity, dt);
  solve_sweep(solver, solver.plan_y, A, B, C, D, sub, 1, own_axis, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  build_z_sweep_constant_property(A, B, C, D, solver.stage_second.data(), q.data(),
                                  W.data(), grid, sub, own_axis, viscosity, dt);
  solve_sweep(solver, solver.plan_z, A, B, C, D, sub, 2, own_axis, i1 * i2, i3);
  equation::scatter_z(delta_out, D, sub);
}

}  // namespace

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

  // 전진 대입 dU → dV → dW (solve_variable_property 1:1 — 물성 자리만 상수)
  adi_sweeps_constant(solver, grid, sub, rhs_u.data(), U, U, V, W, /*own_axis=*/0,
                      viscosity, dt, solver.delta_u.data());
  sub.exchange_halo(solver.delta_u.data());
  close_delta_wall_ghosts(solver, solver.delta_u, 0, sub);

  couple_rhs_v_from_delta_u_constant_property(rhs_v, solver.delta_u, V, grid, sub,
                                              viscosity, dt);
  adi_sweeps_constant(solver, grid, sub, rhs_v.data(), V, U, V, W, /*own_axis=*/1,
                      viscosity, dt, solver.delta_v.data());
  sub.exchange_halo(solver.delta_v.data());
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);

  couple_rhs_w_from_delta_uv_constant_property(rhs_w, solver.delta_u, solver.delta_v,
                                               W, grid, sub, viscosity, dt);
  adi_sweeps_constant(solver, grid, sub, rhs_w.data(), W, U, V, W, /*own_axis=*/2,
                      viscosity, dt, solver.delta_w.data());
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
  equation::add_increment(W, solver.delta_w.data(), sub);
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
                      solver.delta_u.data());
  sub.exchange_halo(solver.delta_u.data());
  close_delta_wall_ghosts(solver, solver.delta_u, 0, sub);

  // ② dV (own y) — M21(δU) 주입 후 풀이
  couple_rhs_v_from_delta_u_variable_property(rhs_v, solver.delta_u, V, viscosity,
                                     inverse_density, grid, sub, Cmu, dt);
  adi_sweeps_variable_property(solver, grid, sub, rhs_v.data(), V, U, V, W, viscosity,
                      inverse_density, /*own_axis=*/1, Cmu, dt,
                      solver.delta_v.data());
  sub.exchange_halo(solver.delta_v.data());
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);

  // ③ dW (own z, 벽) — M31(δU) + M32(δV) 주입 후 풀이
  couple_rhs_w_from_delta_uv_variable_property(rhs_w, solver.delta_u, solver.delta_v, W,
                                      viscosity, inverse_density, grid, sub, Cmu, dt);
  adi_sweeps_variable_property(solver, grid, sub, rhs_w.data(), W, U, V, W, viscosity,
                      inverse_density, /*own_axis=*/2, Cmu, dt,
                      solver.delta_w.data());
  sub.exchange_halo(solver.delta_w.data());
  // 하단 벽면(내부 행 k=h)의 가짜 증분 제거 포함 — 커플링이 소비하기 전
  // (벽 전단 1/dmz 증폭으로 U 스파이크가 되는 경로; 참조 Report/27)
  close_delta_wall_ghosts(solver, solver.delta_w, 2, sub);

  // ── 사후 상삼각: δV ← δW, 동기화, δU ← (보정된 δV) + δW. δW 는 무보정 ──
  block_couple_delta_v_variable_property(solver.delta_v, solver.delta_w, V, viscosity,
                                inverse_density, grid, sub, Cmu, dt);
  sub.exchange_halo(solver.delta_v.data());
  close_delta_wall_ghosts(solver, solver.delta_v, 1, sub);
  block_couple_delta_u_variable_property(solver.delta_u, solver.delta_v, solver.delta_w, U,
                                viscosity, inverse_density, grid, sub, Cmu, dt);

  // 속도 갱신 (내부만) — ghost 재동기화는 호출자의 다음 sync 몫
  equation::add_increment(U, solver.delta_u.data(), sub);
  equation::add_increment(V, solver.delta_v.data(), sub);
  equation::add_increment(W, solver.delta_w.data(), sub);
}

}  // namespace mpmstd::equation::momentum
