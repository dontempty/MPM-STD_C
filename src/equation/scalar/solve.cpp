#include "equation/scalar/solve.hpp"

#include <cstddef>
#include <stdexcept>

#include "common/field_checks.hpp"
#include "equation/scalar/system/system.hpp"
#include "equation/sweep_kernels.hpp"

// 오케스트레이션 전용 TU — CPU/GPU 양쪽 빌드 공용. 루프·커널은
// system.{cpp,cu}(밴드)와 equation/sweep_kernels.{cpp,cu}(산포·갱신)가
// 백엔드별로 구현한다.

namespace mpmstd::equation::scalar {

Solver::Solver(const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  plan_x.create(i2 * i3, sub.topology().x().comm);
  plan_y.create(i1 * i3, sub.topology().y().comm);
  plan_z.create(i1 * i2, sub.topology().z().comm);
  const std::size_t n_band = static_cast<std::size_t>(i1) * i2 * i3;
  band_a.assign(n_band);
  band_b.assign(n_band);
  band_c.assign(n_band);
  band_d.assign(n_band);
  stage_first.assign(sub.n_elements());
  stage_second.assign(sub.n_elements());
}

void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs, field::Field& T, const real_t alpha,
           const real_t dt) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  check_field_matches("scalar::solve", rhs, sub);
  check_field_matches("scalar::solve", T, sub);
  if (solver.band_a.size() != static_cast<std::size_t>(i1) * i2 * i3)
    throw std::invalid_argument(
        "scalar::solve: Solver was created for a different subdomain");

  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();

  // x 스윕 (주기 → 순환 TDMA):  (I − ½dtαLx) δ₁ = rhs
  build_x_system(A, B, C, D, rhs.data(), grid, sub, alpha, dt);
  linear_solver::solve_cyclic_many(solver.plan_x, A, B, C, D, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  // y 스윕 (주기):  (I − ½dtαLy) δ₂ = δ₁
  build_y_system(A, B, C, D, solver.stage_first.data(), grid, sub, alpha, dt);
  linear_solver::solve_cyclic_many(solver.plan_y, A, B, C, D, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  // z 스윕 (벽):  (I − ½dtαLz) δθ = δ₂ — 벽 행은 ghost 정책과 쌍인 fold 로 닫는다
  build_z_system(A, B, C, D, solver.stage_second.data(), grid, sub, alpha, dt);
  fold_z_wall_mirror(A, B, C, sub, i1 * i2, i3);
  linear_solver::solve_many(solver.plan_z, A, B, C, D, i1 * i2, i3);
  equation::scatter_z(solver.stage_first.data(), D, sub);

  // θⁿ⁺¹ = θⁿ + δθ (내부만)
  equation::add_increment(T, solver.stage_first.data(), sub);
}

}  // namespace mpmstd::equation::scalar
