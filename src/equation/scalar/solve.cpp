#include <cstddef>
#include <stdexcept>

// 구간 타이머 (Solver::timer 가 있을 때만; 없으면 no-op) — momentum/solve.cpp 와 같은 틀
#define SCALAR_TIMER_BEGIN(name) \
  do { if (solver.timer) solver.timer->begin(name); } while (0)
#define SCALAR_TIMER_END(name) \
  do { if (solver.timer) solver.timer->end(name); } while (0)

#include "common/field_checks.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/scalar/system/system.hpp"
#include "equation/sweep_kernels.hpp"
#include "equation/band_layout.hpp"
// [최적화 2026-10-18] solve_variable_property: D 밴드 핑퐁(scatter_x/y 제거), scatter_z 가 T += δT 를 바로(δT 버퍼 없음).
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
      solver.wall_closure[a][side] =  // Inflow/Outflow/Continuity/Periodic → 0 (δ_ghost = 0)
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
  real_t* D0 = solver.band_d.data();   // D 핑퐁 (band_layout.hpp): x → D0, y 는 D0 를 읽어 D1, z 는 D1 을 읽어 D0
  real_t* D1 = solver.band_d2.data();
  const stencil::WallMask wall = make_wall_mask(solver, sub);

  // x 스윕 (첫 스윕, 융합 팩): 밴드 + D = dt·(R − ΣM·θⁿ + …)
  SCALAR_TIMER_BEGIN("scalar.build_x");
  build_x_system_variable_property(A, B, C, D0, T.data(), U.data(), V.data(), W.data(),
                          properties, grid, sub, wall, Ct, dt);
  SCALAR_TIMER_END("scalar.build_x");
  SCALAR_TIMER_BEGIN("scalar.tdma_x");
  solve_sweep(solver.plan_x, A, B, C, D0, sub, 0, i2 * i3, i1);
  SCALAR_TIMER_END("scalar.tdma_x");

  // y 스윕: D = x 증분 통과
  SCALAR_TIMER_BEGIN("scalar.build_y");
  build_y_system_variable_property(A, B, C, D1, D0, T.data(), U.data(),
                          V.data(), W.data(), properties, grid, sub, wall, Ct, dt,
                          SourceLayout::BandX);
  SCALAR_TIMER_END("scalar.build_y");
  SCALAR_TIMER_BEGIN("scalar.tdma_y");
  solve_sweep(solver.plan_y, A, B, C, D1, sub, 1, i1 * i3, i2);
  SCALAR_TIMER_END("scalar.tdma_y");

  // z 스윕: 벽 행 닫기 = 조립 안의 dκ̂·비대각 마스크 (fold 호출 없음)
  SCALAR_TIMER_BEGIN("scalar.build_z");
  build_z_system_variable_property(A, B, C, D0, D1, T.data(), U.data(),
                          V.data(), W.data(), properties, grid, sub, wall, Ct, dt,
                          SourceLayout::BandY);
  SCALAR_TIMER_END("scalar.build_z");
  SCALAR_TIMER_BEGIN("scalar.tdma_z");
  solve_sweep(solver.plan_z, A, B, C, D0, sub, 2, i1 * i2, i3);
  SCALAR_TIMER_END("scalar.tdma_z");
  // 산포(타일 전치) + T += δT 를 한 커널로 (δT 버퍼 없음, add_increment 패스 제거)
  SCALAR_TIMER_BEGIN("scalar.scatter_z");
  equation::scatter_z(nullptr, D0, sub, T.data());
  SCALAR_TIMER_END("scalar.scatter_z");

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
  // [최적화 2026-10-18] 가변물성 판과 같은 D 밴드 핑퐁 + scatter_z 가 T += δT 를 바로
  real_t* D0 = solver.band_d.data();
  real_t* D1 = solver.band_d2.data();
  const stencil::WallMask wall = make_wall_mask(solver, sub);

  // x 스윕 (첫 스윕, 융합 팩): 밴드 + D = dt·(R − ΣM·θⁿ + …)
  build_x_system_constant_property(A, B, C, D0, T.data(), U.data(), V.data(),
                                   W.data(), grid, sub, wall, Ct, dt);
  solve_sweep(solver.plan_x, A, B, C, D0, sub, 0, i2 * i3, i1);

  // y 스윕: D = x 증분 통과
  build_y_system_constant_property(A, B, C, D1, D0, T.data(),
                                   U.data(), V.data(), W.data(), grid, sub, wall, Ct, dt,
                                   SourceLayout::BandX);
  solve_sweep(solver.plan_y, A, B, C, D1, sub, 1, i1 * i3, i2);

  // z 스윕: 벽 행 닫기 = 조립 안의 dκ̂·비대각 마스크 (fold 호출 없음)
  build_z_system_constant_property(A, B, C, D0, D1, T.data(),
                                   U.data(), V.data(), W.data(), grid, sub, wall, Ct, dt,
                                   SourceLayout::BandY);
  solve_sweep(solver.plan_z, A, B, C, D0, sub, 2, i1 * i2, i3);
  equation::scatter_z(nullptr, D0, sub, T.data());   // 산포 + T += δT

}

}  // namespace mpmstd::equation::scalar
