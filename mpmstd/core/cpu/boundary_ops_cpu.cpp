#include "core/boundary_ops.hpp"
#include "common/macros.hpp"        // kHaloWidth
#include "boundary/bc_kind.hpp"     // bc_kind_name

#include <mpi.h>
#include <stdexcept>
#include <string>

namespace mpmstd::core {

using boundary::BcKind;
using boundary::GhostPolicy;
using boundary::FaceBc;

namespace {

bool is_global_boundary(const Subdomain& sub, Direction d, Side s) {
  const auto& axis = sub.topology().axis(d);
  return (s == Side::Minus) ? axis.west_rank == MPI_PROC_NULL
                            : axis.east_rank == MPI_PROC_NULL;
}

// Cell-centered Dirichlet ghost: ZeroGhost → ghost=v_wall; Antisymmetric →
// ghost=2*v_wall-interior (velocity uses ZeroGhost; T uses antisymmetric).
void fill_dirichlet(CpuField& phi, Direction d, Side s, const FaceBc& bc, real_t t) {
  const int n1 = phi.n_total(Direction::X), n2 = phi.n_total(Direction::Y), n3 = phi.n_total(Direction::Z);
  const real_t vw = bc.value;
  const bool anti = (bc.ghost_policy == GhostPolicy::Antisymmetric);
  if (d == Direction::X) {
    const int g = (s == Side::Minus) ? 0 : n1 - 1, in = (s == Side::Minus) ? kHaloWidth : n1 - 1 - kHaloWidth;
    for (int j = 0; j < n2; ++j) for (int k = 0; k < n3; ++k)
      phi.at(g, j, k) = anti ? (2 * vw - phi.at(in, j, k)) : vw;
  } else if (d == Direction::Y) {
    const int g = (s == Side::Minus) ? 0 : n2 - 1, in = (s == Side::Minus) ? kHaloWidth : n2 - 1 - kHaloWidth;
    for (int i = 0; i < n1; ++i) for (int k = 0; k < n3; ++k)
      phi.at(i, g, k) = anti ? (2 * vw - phi.at(i, in, k)) : vw;
  } else {
    const int g = (s == Side::Minus) ? 0 : n3 - 1, in = (s == Side::Minus) ? kHaloWidth : n3 - 1 - kHaloWidth;
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j)
      phi.at(i, j, g) = anti ? (2 * vw - phi.at(i, j, in)) : vw;
  }
}

// Cell-centered Neumann: ghost = interior_first +/- v (sign by side).
void fill_neumann(CpuField& phi, Direction d, Side s, const FaceBc& bc, real_t t) {
  const int n1 = phi.n_total(Direction::X), n2 = phi.n_total(Direction::Y), n3 = phi.n_total(Direction::Z);
  const real_t v = bc.value;
  const real_t sgn = (s == Side::Minus) ? real_t{-1} : real_t{1};
  if (d == Direction::X) {
    const int g = (s == Side::Minus) ? 0 : n1 - 1, in = (s == Side::Minus) ? kHaloWidth : n1 - 1 - kHaloWidth;
    for (int j = 0; j < n2; ++j) for (int k = 0; k < n3; ++k) phi.at(g, j, k) = phi.at(in, j, k) + sgn * v;
  } else if (d == Direction::Y) {
    const int g = (s == Side::Minus) ? 0 : n2 - 1, in = (s == Side::Minus) ? kHaloWidth : n2 - 1 - kHaloWidth;
    for (int i = 0; i < n1; ++i) for (int k = 0; k < n3; ++k) phi.at(i, g, k) = phi.at(i, in, k) + sgn * v;
  } else {
    const int g = (s == Side::Minus) ? 0 : n3 - 1, in = (s == Side::Minus) ? kHaloWidth : n3 - 1 - kHaloWidth;
    for (int i = 0; i < n1; ++i) for (int j = 0; j < n2; ++j) phi.at(i, j, g) = phi.at(i, j, in) + sgn * v;
  }
}

} // anonymous namespace

void apply_ghost_cpu(CpuField& phi, const FieldBoundary& fbc, const Subdomain& sub, real_t t) {
  for (int a = 0; a < 3; ++a) {
    const Direction d = static_cast<Direction>(a);
    for (Side s : {Side::Minus, Side::Plus}) {
      if (!is_global_boundary(sub, d, s)) continue;   // interior interface → halo exchange
      const FaceBc& bc = fbc.face(d, s);
      switch (bc.kind) {
        case BcKind::Periodic:  break;                 // halo wrap handles it
        case BcKind::Dirichlet: fill_dirichlet(phi, d, s, bc, t); break;
        case BcKind::Neumann:   fill_neumann(phi, d, s, bc, t);   break;
        default:
          throw std::runtime_error(std::string("apply_ghost_cpu: BcKind not implemented: ")
                                   + boundary::bc_kind_name(bc.kind));
      }
    }
  }
}

void modify_tdma_row_cpu(Direction wall_axis, const FieldBoundary& fbc,
                         const parallel::mpi::CartComm1D& axis_comm,
                         real_t* A, real_t* B, real_t* C, real_t* /*D*/,
                         int n_sys, int n_row) {
  const bool owns_lower = (axis_comm.west_rank == MPI_PROC_NULL);
  const bool owns_upper = (axis_comm.east_rank == MPI_PROC_NULL);

  auto fold_lower = [&](const FaceBc& face) {
    for (int s = 0; s < n_sys; ++s) {
      const int p = s;  // row 0
      if (face.kind == BcKind::Dirichlet || face.kind == BcKind::Neumann) {
        B[p] += A[p]; A[p] = real_t{0};   // δ_ghost = ±δ_interior fold (see original)
      } else {
        throw std::runtime_error(std::string("modify_tdma_row_cpu(lower): BcKind not implemented: ")
                                 + boundary::bc_kind_name(face.kind));
      }
    }
  };
  auto fold_upper = [&](const FaceBc& face) {
    for (int s = 0; s < n_sys; ++s) {
      const int p = (n_row - 1) * n_sys + s;
      if (face.kind == BcKind::Dirichlet || face.kind == BcKind::Neumann) {
        B[p] += C[p]; C[p] = real_t{0};
      } else {
        throw std::runtime_error(std::string("modify_tdma_row_cpu(upper): BcKind not implemented: ")
                                 + boundary::bc_kind_name(face.kind));
      }
    }
  };

  if (owns_lower) fold_lower(fbc.face(wall_axis, Side::Minus));
  if (owns_upper) fold_upper(fbc.face(wall_axis, Side::Plus));
}

} // namespace mpmstd::core
