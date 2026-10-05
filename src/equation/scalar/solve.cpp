#include <cstddef>
#include <stdexcept>

#include "common/field_checks.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/scalar/system/system.hpp"
#include "equation/sweep_kernels.hpp"
#include <mpi.h>

// 가변-κ̂ (NOB) 에너지 solve — 오케스트레이션 전용 TU (CPU/GPU 공용).
// OB solve 와 달리 rhs 인자가 없다: 첫 스윕(x)이 우변을 융합 조립한다
// (system.hpp 머리말). z 벽 닫기는 조립 마스크가 담당하므로 OB 의
// fold_z_wall_mirror 를 부르지 않는다.

namespace mpmstd::equation::scalar {

namespace {
// 주기 축 → 순환 TDMA, 벽 축 → 비주기 TDMA (벽 행 닫기는 조립 마스크가 이미 했다)
void solve_sweep(linear_solver::TdmaPlan& plan, real_t* A, real_t* B, real_t* C,
                 real_t* D, const parallel::Subdomain& sub, const int axis,
                 const int nsys, const int nrow) {
  if (sub.topology().axis(axis).periodic)
    linear_solver::solve_cyclic_many(plan, A, B, C, D, nsys, nrow);
  else
    linear_solver::solve_many(plan, A, B, C, D, nsys, nrow);
}

// 세 축의 벽 소유 + 닫기 계수 (주기 축은 소유 false — 기존 결과와 비트 동일)
stencil::WallMask make_wall_mask(const Solver& solver, const parallel::Subdomain& sub) {
  stencil::WallMask m;
  for (int a = 0; a < 3; ++a) {
    m.lo[a] = sub.topology().axis(a).minus_neighbor == MPI_PROC_NULL;
    m.hi[a] = sub.topology().axis(a).plus_neighbor == MPI_PROC_NULL;
    m.closure_lo[a] = solver.wall_closure[a][0];
    m.closure_hi[a] = solver.wall_closure[a][1];
  }
  return m;
}
}  // namespace

void set_wall_closure(Solver& solver, const boundary::BoundaryCondition& bc,
                      const boundary::Field which) {
  for (int a = 0; a < 3; ++a)
    for (int side = 0; side < 2; ++side) {
      const boundary::BcKind kind = bc.at(which, a, side).kind;
      solver.wall_closure[a][side] =
          kind == boundary::BcKind::Dirichlet ? real_t(-1)
          : kind == boundary::BcKind::Neumann ? real_t(1) : real_t(0);
    }
}

void solve_variable_property(Solver& solver, const grid::Grid& grid,
                    const parallel::Subdomain& sub, field::Field& T,
                    const field::Field& U, const field::Field& V,
                    const field::Field& W, const physics::PropertyFields& properties,
                    const real_t Ct, const real_t dt) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  check_field_matches("scalar::solve_variable_property", T, sub);
  check_field_matches("scalar::solve_variable_property", U, sub);
  check_field_matches("scalar::solve_variable_property", V, sub);
  check_field_matches("scalar::solve_variable_property", W, sub);
  check_field_matches("scalar::solve_variable_property", properties.conductivity, sub);
  if (solver.band_a.size() != static_cast<std::size_t>(i1) * i2 * i3)
    throw std::invalid_argument(
        "scalar::solve_variable_property: Solver was created for a different subdomain");

  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();
  const stencil::WallMask wall = make_wall_mask(solver, sub);

  // x 스윕 (첫 스윕, 융합 팩): 밴드 + D = dt·(R − ΣM·θⁿ + …)
  build_x_system_variable_property(A, B, C, D, T.data(), U.data(), V.data(), W.data(),
                          properties, grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_x, A, B, C, D, sub, 0, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  // y 스윕: D = x 증분 통과
  build_y_system_variable_property(A, B, C, D, solver.stage_first.data(), T.data(), U.data(),
                          V.data(), W.data(), properties, grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_y, A, B, C, D, sub, 1, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  // z 스윕: 벽 행 닫기 = 조립 안의 dκ̂·비대각 마스크 (fold 호출 없음)
  build_z_system_variable_property(A, B, C, D, solver.stage_second.data(), T.data(), U.data(),
                          V.data(), W.data(), properties, grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_z, A, B, C, D, sub, 2, i1 * i2, i3);
  equation::scatter_z(solver.stage_first.data(), D, sub);

  // θⁿ⁺¹ = θⁿ + δθ (내부만; ghost 재동기화는 호출자 몫)
  equation::add_increment(T, solver.stage_first.data(), sub);
}

// ── 상수물성 판 (plan §18-1) — 위와 동일 구조, 물성 배열 없음 ──────────────
void solve_constant_property(Solver& solver, const grid::Grid& grid,
                    const parallel::Subdomain& sub, field::Field& T,
                    const field::Field& U, const field::Field& V,
                    const field::Field& W, const real_t Ct, const real_t dt) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  check_field_matches("scalar::solve_constant_property", T, sub);
  check_field_matches("scalar::solve_constant_property", U, sub);
  check_field_matches("scalar::solve_constant_property", V, sub);
  check_field_matches("scalar::solve_constant_property", W, sub);
  if (solver.band_a.size() != static_cast<std::size_t>(i1) * i2 * i3)
    throw std::invalid_argument(
        "scalar::solve_constant_property: Solver was created for a different subdomain");

  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();
  const stencil::WallMask wall = make_wall_mask(solver, sub);

  // x 스윕 (첫 스윕, 융합 팩): 밴드 + D = dt·(R − ΣM·θⁿ + …)
  build_x_system_constant_property(A, B, C, D, T.data(), U.data(), V.data(),
                                   W.data(), grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_x, A, B, C, D, sub, 0, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  // y 스윕: D = x 증분 통과
  build_y_system_constant_property(A, B, C, D, solver.stage_first.data(), T.data(),
                                   U.data(), V.data(), W.data(), grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_y, A, B, C, D, sub, 1, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  // z 스윕: 벽 행 닫기 = 조립 안의 dκ̂·비대각 마스크 (fold 호출 없음)
  build_z_system_constant_property(A, B, C, D, solver.stage_second.data(), T.data(),
                                   U.data(), V.data(), W.data(), grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_z, A, B, C, D, sub, 2, i1 * i2, i3);
  equation::scatter_z(solver.stage_first.data(), D, sub);

  // θⁿ⁺¹ = θⁿ + δθ (내부만; ghost 재동기화는 호출자 몫)
  equation::add_increment(T, solver.stage_first.data(), sub);
}

}  // namespace mpmstd::equation::scalar
