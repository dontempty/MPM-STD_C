#include "boundary/apply.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "boundary/wall.hpp"

namespace mpmstd::boundary {

namespace {

// 셀 중심 필드 한 벽면: Dirichlet → mirror 또는 value, Neumann → zero_gradient.
void fill_cell_centered_face(field::Field& f, const FaceBc& face, const int axis,
                             const int side, const parallel::Subdomain& sub,
                             const bool dirichlet_as_mirror) {
  if (face.kind == BcKind::Dirichlet) {
    if (dirichlet_as_mirror) fill_wall_mirror(f, sub, axis, side, face.value);
    else fill_wall_value(f, sub, axis, side, face.value);
  } else {
    fill_wall_zero_gradient(f, sub, axis, side);
  }
}

}  // namespace

void apply_velocity_wall_ghosts(field::Field& U, field::Field& V, field::Field& W,
                                const BoundaryCondition& bc,
                                const parallel::Subdomain& sub) {
  field::Field* component[3] = {&U, &V, &W};
  const Field row[3] = {Field::U, Field::V, Field::W};
  for (int axis = 0; axis < 3; ++axis) {
    if (bc.is_periodic(axis)) continue;
    for (int side = 0; side < 2; ++side)
      for (int c = 0; c < 3; ++c) {
        const FaceBc& face = bc.at(row[c], axis, side);
        // 열린 경계(유입 프로파일·유출 대류·연속식)는 boundary/open 이 앱 호출로 채운다
        if (face.kind == BcKind::Inflow || face.kind == BcKind::Outflow
            || face.kind == BcKind::Continuity || face.kind == BcKind::FreeStream)
          continue;
        if (c == axis) {
          // 법선 성분: 벽 면 자체. 불투과(0)만 지원 — validate 가 Dirichlet 은 보장.
          if (face.value != 0)
            throw std::invalid_argument(
                "apply_velocity_wall_ghosts: wall-normal velocity must be 0 (axis "
                + std::to_string(axis) + ", side " + std::to_string(side) + ")");
          zero_wall_faces(*component[c], sub, axis, side);
        } else {
          // 접선 성분: Dirichlet → value(기본, 1차) 또는 mirror(2차; boundary_condition.hpp
          // VelocityGhost — 운동량 닫기와 짝), Neumann → zero_gradient(slip).
          fill_cell_centered_face(
              *component[c], face, axis, side, sub,
              /*dirichlet_as_mirror=*/bc.velocity_ghost == VelocityGhost::Mirror);
        }
      }
  }
}

void apply_scalar_wall_ghosts(field::Field& f, const Field which,
                              const BoundaryCondition& bc, const parallel::Subdomain& sub) {
  for (int axis = 0; axis < 3; ++axis) {
    if (bc.is_periodic(axis)) continue;
    for (int side = 0; side < 2; ++side) {
      const FaceBc& face = bc.at(which, axis, side);
      if (face.kind == BcKind::Inflow || face.kind == BcKind::Outflow
          || face.kind == BcKind::Continuity)
        continue;  // 열린 경계는 앱이 (boundary/open)
      fill_cell_centered_face(f, face, axis, side, sub, /*dirichlet_as_mirror=*/true);
    }
  }
}

void apply_pressure_wall_ghosts(field::Field& P, const BoundaryCondition& bc,
                                const parallel::Subdomain& sub) {
  for (int axis = 0; axis < 3; ++axis) {
    if (bc.is_periodic(axis)) continue;
    for (int side = 0; side < 2; ++side) {
      if (bc.at(Field::P, axis, side).kind != BcKind::Neumann)
        throw std::invalid_argument("apply_pressure_wall_ghosts: pressure must be Neumann");
      fill_wall_zero_gradient(P, sub, axis, side);
    }
  }
}

}  // namespace mpmstd::boundary
