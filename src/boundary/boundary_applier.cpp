#include "boundary/boundary_applier.hpp"
#include "common/macros.hpp"

#include <mpi.h>
#include <stdexcept>
#include <string>

namespace mpmstd::boundary {

namespace {

// True iff this rank holds the global boundary plane on side `s` of axis `d`.
// We detect it by inspecting the axis 1D sub-communicator's neighbor: a
// neighbor of MPI_PROC_NULL means there is no rank beyond us on that side.
bool is_global_boundary(const field::ScalarField& phi, Direction d, Side s) {
  const auto& axis = phi.subdomain().topology().axis(d);
  if (s == Side::Minus) return axis.west_rank == MPI_PROC_NULL;
  return axis.east_rank == MPI_PROC_NULL;
}

} // anonymous namespace

void BoundaryApplier::apply_ghost(field::ScalarField& phi,
                                    const FieldBoundary& fbc,
                                    real_t t) const {
  for (int a = 0; a < 3; ++a) {
    const Direction d = static_cast<Direction>(a);
    for (Side s : { Side::Minus, Side::Plus }) {
      if (!is_global_boundary(phi, d, s)) {
        // Interior interface between two ranks: MPI halo exchange handles it.
        continue;
      }
      const FaceBc& bc = fbc.face(d, s);
      switch (bc.kind) {
        case BcKind::Periodic:
          fill_face_periodic_(phi, d, s, bc, t);
          break;
        case BcKind::Dirichlet:
          fill_face_dirichlet_(phi, d, s, bc, t);
          break;
        case BcKind::Neumann:
          fill_face_neumann_(phi, d, s, bc, t);
          break;
        case BcKind::Wall:
        case BcKind::Inflow:
        case BcKind::Outflow:
          throw std::runtime_error(
            std::string("BoundaryApplier::apply_ghost: BcKind '") +
            bc_kind_name(bc.kind) + "' not yet implemented");
      }
    }
  }
}

// ---- per-face fillers ------------------------------------------------------

void BoundaryApplier::fill_face_periodic_(field::ScalarField& /*phi*/,
                                            Direction /*d*/, Side /*s*/,
                                            const FaceBc& /*bc*/,
                                            real_t /*t*/) const {
  // Periodic global boundaries are filled by the MPI halo exchange (the axis
  // sub-comm wraps around when MPI Cart was created with `periods[d]=1`), so
  // here we have nothing to do.  If a rank ever sees Periodic on a *global*
  // boundary face (i.e. its neighbor is MPI_PROC_NULL on that axis), that
  // would indicate a topology/BC mismatch; we rely on Problem::validate() to
  // catch that case during setup.
}

void BoundaryApplier::fill_face_dirichlet_(field::ScalarField& phi,
                                             Direction d, Side s,
                                             const FaceBc& bc,
                                             real_t t) const {
  // Cell-centered Dirichlet ghost fill.  Two policies:
  //
  //   ZeroGhost:     ghost = v_wall                   (1st-order)
  //   Antisymmetric: ghost = 2*v_wall - phi_interior  (2nd-order)
  //
  // ZeroGhost is used for velocity (U/V/W) — antisymmetric ghost would fold
  // extra ν/dz² damping into the implicit operator and break sub-critical
  // transition.  Antisymmetric is used for temperature (T) where 2nd-order
  // wall accuracy matters and no such physical issue exists.
  //
  // The matching modify_tdma_row call must be consistent: ZeroGhost → A=0
  // only; Antisymmetric → B -= A then A=0 (fold δ_ghost = -δ_interior).
  const int n1 = phi.n_total(Direction::X);
  const int n2 = phi.n_total(Direction::Y);
  const int n3 = phi.n_total(Direction::Z);

  const real_t v_wall = bc.value;
  const bool antisymm = (bc.ghost_policy == GhostPolicy::Antisymmetric);

  if (d == Direction::X) {
    const int ghost    = (s == Side::Minus) ? 0           : (n1 - 1);
    const int interior = (s == Side::Minus) ? kHaloWidth  : (n1 - 1 - kHaloWidth);
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        phi.host_at(ghost, j, k) = antisymm
            ? (2.0 * v_wall - phi.host_at(interior, j, k))
            : v_wall;
  } else if (d == Direction::Y) {
    const int ghost    = (s == Side::Minus) ? 0           : (n2 - 1);
    const int interior = (s == Side::Minus) ? kHaloWidth  : (n2 - 1 - kHaloWidth);
    for (int i = 0; i < n1; ++i)
      for (int k = 0; k < n3; ++k)
        phi.host_at(i, ghost, k) = antisymm
            ? (2.0 * v_wall - phi.host_at(i, interior, k))
            : v_wall;
  } else { // Direction::Z
    const int ghost    = (s == Side::Minus) ? 0           : (n3 - 1);
    const int interior = (s == Side::Minus) ? kHaloWidth  : (n3 - 1 - kHaloWidth);
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        phi.host_at(i, j, ghost) = antisymm
            ? (2.0 * v_wall - phi.host_at(i, j, interior))
            : v_wall;
  }
}

void BoundaryApplier::fill_face_neumann_(field::ScalarField& phi,
                                           Direction d, Side s,
                                           const FaceBc& bc,
                                           real_t t) const {
  // Cell-centered Neumann (zero-gradient when bc.value == 0):
  //   ghost = interior_first
  //
  // For non-zero Neumann (∂φ/∂n = v), the precise stencil depends on grid
  // spacing at the wall face.  Phase-1 uses the constant-spacing form
  //   ghost = interior_first  +/-  dx * v
  // but we leave the dx-dependent term for later — currently we apply
  //   ghost = interior_first - sign * v
  // (multiplicative dx is incorporated by the equation using this field).
  const int n1 = phi.n_total(Direction::X);
  const int n2 = phi.n_total(Direction::Y);
  const int n3 = phi.n_total(Direction::Z);

  const real_t v = bc.value;

  auto copy_face = [&](int ghost_i, int interior_i, Direction axis) {
    const real_t sign = (s == Side::Minus) ? -1.0 : +1.0;
    if (axis == Direction::X) {
      for (int j = 0; j < n2; ++j)
        for (int k = 0; k < n3; ++k)
          phi.host_at(ghost_i, j, k) =
              phi.host_at(interior_i, j, k) + sign * v;
    } else if (axis == Direction::Y) {
      for (int i = 0; i < n1; ++i)
        for (int k = 0; k < n3; ++k)
          phi.host_at(i, ghost_i, k) =
              phi.host_at(i, interior_i, k) + sign * v;
    } else {
      for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
          phi.host_at(i, j, ghost_i) =
              phi.host_at(i, j, interior_i) + sign * v;
    }
  };

  if (d == Direction::X) {
    const int ghost    = (s == Side::Minus) ? 0           : (n1 - 1);
    const int interior = (s == Side::Minus) ? kHaloWidth  : (n1 - 1 - kHaloWidth);
    copy_face(ghost, interior, Direction::X);
  } else if (d == Direction::Y) {
    const int ghost    = (s == Side::Minus) ? 0           : (n2 - 1);
    const int interior = (s == Side::Minus) ? kHaloWidth  : (n2 - 1 - kHaloWidth);
    copy_face(ghost, interior, Direction::Y);
  } else {
    const int ghost    = (s == Side::Minus) ? 0           : (n3 - 1);
    const int interior = (s == Side::Minus) ? kHaloWidth  : (n3 - 1 - kHaloWidth);
    copy_face(ghost, interior, Direction::Z);
  }
}

// ---- TDMA row modification (cell-centered, M2) ----------------------------
//
// At the first interior cell along the wall axis, the discrete Laplacian
// uses a ghost cell. When we're solving for the INCREMENT δ (Douglas ADI):
//
//   * Dirichlet wall (wall value fixed in time)
//        →  δ_ghost = 0
//        →  A[row 0] * δ_ghost = 0
//        →  set A[row 0] = 0          (D unchanged — explicit RHS already
//                                       absorbed the wall value via the
//                                       ghost layer set by apply_ghost.)
//
//   * Neumann wall (zero-gradient ⇒ ghost = first interior; same for δ)
//        →  δ_ghost = δ_first_interior
//        →  A[row 0] * δ_ghost = A[row 0] * δ_first_interior
//        →  fold into diagonal:  B[row 0] += A[row 0],  A[row 0] = 0.
//
// Mirror logic at the upper wall (row n_row - 1, C[]).

void BoundaryApplier::modify_tdma_row(Direction wall_axis,
                                        const FieldBoundary& fbc,
                                        const parallel::mpi::CartComm1D& axis_comm,
                                        real_t* A, real_t* B, real_t* C, real_t* /*D*/,
                                        int n_sys, int n_row) const {
  const bool owns_lower = (axis_comm.west_rank == MPI_PROC_NULL);
  const bool owns_upper = (axis_comm.east_rank == MPI_PROC_NULL);

  // Dirichlet ghost policy determines how the sub/super-diagonal is handled:
  //
  //   ZeroGhost:     δ_ghost = 0   → just drop A (set to 0); D unchanged.
  //   Antisymmetric: δ_ghost = -δ_interior  (since ghost = 2v - φ)
  //                  → fold: B -= A, then A = 0  (and B -= C, C = 0 for upper).
  //
  // Neumann: ghost = interior → δ_ghost = δ_interior → fold: B += A, A = 0.

  auto modify_lower_row = [&](const FaceBc& face) {
    for (int s = 0; s < n_sys; ++s) {
      const int p = s;                       // row 0, column s
      switch (face.kind) {
        case BcKind::Dirichlet:
          // Neumann-style fold: B += A, A = 0.
          // This gives B+C = 1 at the wall row, so the discrete steady-state
          // Laplacian matches the continuous -F/nu (Poiseuille condition).
          // Flag-drop (A=0 only) gives B+C = 1+|av|, which biases the wall-cell
          // velocity and produces a wrong WSS fixed point (~16x laminar).
          // The antisymmetric ghost in apply_ghost already gives the correct
          // explicit Laplacian; the fold here only affects the implicit delta system.
          B[p] += A[p];
          A[p]  = 0.0;
          break;
        case BcKind::Neumann:
          B[p] += A[p];
          A[p]  = 0.0;
          break;
        case BcKind::Periodic:
          throw std::runtime_error(
            "BoundaryApplier::modify_tdma_row: Periodic on wall axis is "
            "inconsistent (use solve_many_cyclic instead).");
        default:
          throw std::runtime_error(
            std::string("BoundaryApplier::modify_tdma_row: BcKind '") +
            bc_kind_name(face.kind) + "' not implemented yet");
      }
    }
  };
  auto modify_upper_row = [&](const FaceBc& face) {
    for (int s = 0; s < n_sys; ++s) {
      const int p = (n_row - 1) * n_sys + s;
      switch (face.kind) {
        case BcKind::Dirichlet:
          // Neumann-style fold (same reasoning as lower wall).
          B[p] += C[p];
          C[p]  = 0.0;
          break;
        case BcKind::Neumann:
          B[p] += C[p];
          C[p]  = 0.0;
          break;
        case BcKind::Periodic:
          throw std::runtime_error(
            "BoundaryApplier::modify_tdma_row: Periodic on wall axis is "
            "inconsistent (use solve_many_cyclic instead).");
        default:
          throw std::runtime_error(
            std::string("BoundaryApplier::modify_tdma_row: BcKind '") +
            bc_kind_name(face.kind) + "' not implemented yet");
      }
    }
  };

  if (owns_lower) modify_lower_row(fbc.face(wall_axis, Side::Minus));
  if (owns_upper) modify_upper_row(fbc.face(wall_axis, Side::Plus ));
}

} // namespace mpmstd::boundary
