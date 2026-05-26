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
  // Cell-centered Dirichlet: we set the ghost cell directly to the boundary
  // value.  Combined with `modify_tdma_row` (M3+), this implements the
  // zero-ghost + matrix-flag-drop policy on the wall row.
  //
  // Note: for cell-centered fields whose physical wall sits on the face
  // exactly between the last interior cell and the ghost cell, a more
  // accurate stencil is ghost = 2*v_wall - interior.  Choosing one or the
  // other is the responsibility of the equation that owns this field; for
  // now we apply the simple "ghost = v_wall" policy (consistent with the
  // RBC defaults of `U=V=W=0` and `T=±0.5`).
  const int n1 = phi.n_total(Direction::X);
  const int n2 = phi.n_total(Direction::Y);
  const int n3 = phi.n_total(Direction::Z);

  // For x, y, z coordinates we use cell-center positions of the ghost slab;
  // these are passed to the user's value function (constant BCs ignore them).
  const auto& g_xc = phi.subdomain();  // not actually a grid here — we don't
  (void)g_xc;
  // Phase-1: pass (0, 0, 0) — value functions in M1 are all constants.

  if (d == Direction::X) {
    const int i = (s == Side::Minus) ? 0 : (n1 - 1);
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        phi.host_at(i, j, k) = bc.value(0.0, 0.0, 0.0, t);
  } else if (d == Direction::Y) {
    const int j = (s == Side::Minus) ? 0 : (n2 - 1);
    for (int i = 0; i < n1; ++i)
      for (int k = 0; k < n3; ++k)
        phi.host_at(i, j, k) = bc.value(0.0, 0.0, 0.0, t);
  } else { // Direction::Z
    const int k = (s == Side::Minus) ? 0 : (n3 - 1);
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        phi.host_at(i, j, k) = bc.value(0.0, 0.0, 0.0, t);
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

  const real_t v = bc.value(0.0, 0.0, 0.0, t);

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

  auto modify_lower_row = [&](BcKind kind) {
    for (int s = 0; s < n_sys; ++s) {
      const int p = s;                       // row 0, column s
      switch (kind) {
        case BcKind::Dirichlet:
          A[p] = 0.0;
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
            bc_kind_name(kind) + "' not implemented yet");
      }
    }
  };
  auto modify_upper_row = [&](BcKind kind) {
    for (int s = 0; s < n_sys; ++s) {
      const int p = (n_row - 1) * n_sys + s;
      switch (kind) {
        case BcKind::Dirichlet:
          C[p] = 0.0;
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
            bc_kind_name(kind) + "' not implemented yet");
      }
    }
  };

  if (owns_lower) modify_lower_row(fbc.face(wall_axis, Side::Minus).kind);
  if (owns_upper) modify_upper_row(fbc.face(wall_axis, Side::Plus ).kind);
}

} // namespace mpmstd::boundary
