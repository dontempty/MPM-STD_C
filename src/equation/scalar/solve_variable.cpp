#include <cstddef>
#include <stdexcept>

#include "common/field_checks.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/scalar/system/system_variable.hpp"
#include "equation/sweep_kernels.hpp"

// 가변-κ̂ (NOB) 에너지 solve — 오케스트레이션 전용 TU (CPU/GPU 공용).
// OB solve 와 달리 rhs 인자가 없다: 첫 스윕(x)이 우변을 융합 조립한다
// (system_variable.hpp 머리말). z 벽 닫기는 조립 마스크가 담당하므로 OB 의
// fold_z_wall_mirror 를 부르지 않는다.

namespace mpmstd::equation::scalar {

void solve_variable(Solver& solver, const grid::Grid& grid,
                    const parallel::Subdomain& sub, field::Field& T,
                    const field::Field& U, const field::Field& V,
                    const field::Field& W, const physics::PropertyFields& properties,
                    const real_t Ct, const real_t dt) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  check_field_matches("scalar::solve_variable", T, sub);
  check_field_matches("scalar::solve_variable", U, sub);
  check_field_matches("scalar::solve_variable", V, sub);
  check_field_matches("scalar::solve_variable", W, sub);
  check_field_matches("scalar::solve_variable", properties.conductivity, sub);
  if (solver.band_a.size() != static_cast<std::size_t>(i1) * i2 * i3)
    throw std::invalid_argument(
        "scalar::solve_variable: Solver was created for a different subdomain");

  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();

  // x 스윕 (주기, 융합 팩): 밴드 + D = dt·(R − ΣM·θⁿ + …)
  build_x_system_variable(A, B, C, D, T.data(), U.data(), V.data(), W.data(),
                          properties, grid, sub, Ct, dt);
  linear_solver::solve_cyclic_many(solver.plan_x, A, B, C, D, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  // y 스윕 (주기): D = x 증분 통과
  build_y_system_variable(A, B, C, D, solver.stage_first.data(), T.data(), U.data(),
                          V.data(), W.data(), properties, grid, sub, Ct, dt);
  linear_solver::solve_cyclic_many(solver.plan_y, A, B, C, D, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  // z 스윕 (벽): 벽 행 닫기 = 조립 안의 dκ̂·비대각 마스크 (fold 호출 없음)
  build_z_system_variable(A, B, C, D, solver.stage_second.data(), T.data(), U.data(),
                          V.data(), W.data(), properties, grid, sub, Ct, dt);
  linear_solver::solve_many(solver.plan_z, A, B, C, D, i1 * i2, i3);
  equation::scatter_z(solver.stage_first.data(), D, sub);

  // θⁿ⁺¹ = θⁿ + δθ (내부만; ghost 재동기화는 호출자 몫)
  equation::add_increment(T, solver.stage_first.data(), sub);
}

}  // namespace mpmstd::equation::scalar
