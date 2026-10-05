#include "equation/momentum/solve.hpp"

#include <cstddef>
#include <stdexcept>

#include "boundary/wall.hpp"
#include "common/field_checks.hpp"
#include "equation/momentum/solve_kernels.hpp"
#include "equation/sweep_kernels.hpp"
#include "equation/momentum/system/system.hpp"
#include "equation/momentum/wall_rows.hpp"

// 오케스트레이션 전용 TU — CPU/GPU 양쪽 빌드에 그대로 들어간다. 루프·커널은
// solve_kernels.{cpp,cu} 와 system.{cpp,cu} 가 백엔드별로 구현한다.

namespace mpmstd::equation::momentum {

namespace {

// 한 성분의 3-스윕 ADI. z 스윕의 메트릭·벽 처리만 성분에 따라 갈린다.
void adi_sweeps(Solver& solver, const grid::Grid& grid,
                const parallel::Subdomain& sub, const real_t* rhs_data,
                const field::Field& U, const field::Field& V, const field::Field& W,
                const real_t viscosity, const real_t dt, const bool w_is_z_face,
                real_t* delta_out) {
  const int i1 = sub.x().n_local;
  const int i2 = sub.y().n_local;
  const int i3 = sub.z().n_local;
  real_t* A = solver.band_a.data();
  real_t* B = solver.band_b.data();
  real_t* C = solver.band_c.data();
  real_t* D = solver.band_d.data();

  // x 스윕 (주기, 이송 = U)
  build_x_system(A, B, C, D, rhs_data, U, grid, sub, viscosity, dt);
  linear_solver::solve_cyclic_many(solver.plan_x, A, B, C, D, i2 * i3, i1);
  equation::scatter_x(solver.stage_first.data(), D, sub);

  // y 스윕 (주기, 이송 = V)
  build_y_system(A, B, C, D, solver.stage_first.data(), V, grid, sub, viscosity, dt);
  linear_solver::solve_cyclic_many(solver.plan_y, A, B, C, D, i1 * i3, i2);
  equation::scatter_y(solver.stage_second.data(), D, sub);

  // z 스윕 (벽, 이송 = W) — 벽 닫기는 ghost 정책과 쌍 (system.hpp ⚠)
  if (w_is_z_face) {
    build_z_system_z_face(A, B, C, D, solver.stage_second.data(), W, grid, sub,
                          viscosity, dt);
    fold_z_wall_value(A, B, C, sub, i1 * i2, i3);          // 상단: ghost 슬롯의 면 δ=0
    pin_z_bottom_wall_face_row(A, B, C, D, sub, i1 * i2);  // 하단 벽면 = 내부 행 0
  } else {
    build_z_system_cell(A, B, C, D, solver.stage_second.data(), W, grid, sub,
                        viscosity, dt);
    fold_z_wall_value(A, B, C, sub, i1 * i2, i3);
  }
  linear_solver::solve_many(solver.plan_z, A, B, C, D, i1 * i2, i3);
  equation::scatter_z(delta_out, D, sub);
}

}  // namespace

Solver::Solver(const parallel::Subdomain& sub)
    : delta_u(sub.x().n_total, sub.y().n_total, sub.z().n_total, "dU"),
      delta_v(sub.x().n_total, sub.y().n_total, sub.z().n_total, "dV"),
      delta_w(sub.x().n_total, sub.y().n_total, sub.z().n_total, "dW") {
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
           const field::Field& rhs_u, const field::Field& rhs_v,
           const field::Field& rhs_w, field::Field& U, field::Field& V,
           field::Field& W, const real_t viscosity, const real_t dt) {
  check_field_matches("momentum::solve", rhs_u, sub);
  check_field_matches("momentum::solve", rhs_v, sub);
  check_field_matches("momentum::solve", rhs_w, sub);
  if (solver.delta_u.size() != rhs_u.size())
    throw std::invalid_argument(
        "momentum::solve: Solver was created for a different subdomain");

  // ① 성분별 독립 ADI (OB 경로: 전진 대입 없음 — ref_03 §1.5)
  adi_sweeps(solver, grid, sub, rhs_u.data(), U, V, W, viscosity, dt,
             /*w_is_z_face=*/false, solver.delta_u.data());
  adi_sweeps(solver, grid, sub, rhs_v.data(), U, V, W, viscosity, dt,
             /*w_is_z_face=*/false, solver.delta_v.data());
  adi_sweeps(solver, grid, sub, rhs_w.data(), U, V, W, viscosity, dt,
             /*w_is_z_face=*/true, solver.delta_w.data());

  // ② 증분 동기화 — 커플링 스텐실이 이웃 증분을 읽는다.
  //    벽 ghost 정책 = "경계값의 증분은 0" (상수 BC 의 δ).
  sub.exchange_halo(solver.delta_u.data());
  boundary::fill_z_wall_value(solver.delta_u, sub, 0.0, 0.0);
  sub.exchange_halo(solver.delta_v.data());
  boundary::fill_z_wall_value(solver.delta_v, sub, 0.0, 0.0);
  sub.exchange_halo(solver.delta_w.data());
  boundary::zero_z_wall_faces(solver.delta_w, sub);

  // ③ 블록 커플링: δV ← δW, 동기화, δU ← (보정된 δV) + δW. δW 는 무보정.
  block_couple_delta_v(solver.delta_v, solver.delta_w, V, grid, sub, dt);
  sub.exchange_halo(solver.delta_v.data());
  boundary::fill_z_wall_value(solver.delta_v, sub, 0.0, 0.0);
  block_couple_delta_u(solver.delta_u, solver.delta_v, solver.delta_w, U, grid, sub,
                       dt);

  // ④ 속도 갱신 (내부만) — ghost 재동기화는 호출자의 다음 sync 가 맡는다.
  equation::add_increment(U, solver.delta_u.data(), sub);
  equation::add_increment(V, solver.delta_v.data(), sub);
  equation::add_increment(W, solver.delta_w.data(), sub);
}

}  // namespace mpmstd::equation::momentum
