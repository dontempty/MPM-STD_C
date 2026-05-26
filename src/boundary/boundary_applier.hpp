#pragma once

#include "boundary/problem.hpp"
#include "field/scalar_field.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#include "common/types.hpp"

namespace mpmstd::boundary {

// BoundaryApplier interprets a FieldBoundary and acts on field data /
// tridiagonal-matrix coefficients.
//
// Two responsibilities:
//
//   apply_ghost(...)       Fills halo cells on **global** boundary faces
//                          (faces where this rank has MPI_PROC_NULL on the
//                          axis sub-comm). Periodic, Dirichlet, Neumann are
//                          implemented; Wall, Inflow, Outflow throw.
//
//   modify_tdma_row(...)   Amends one ADI-stage tridiagonal system so the
//                          wall BC is enforced.  Cell-centered Dirichlet /
//                          Neumann are implemented in M2.  The face-centered
//                          variant used by momentum solver arrives in M3.
//
//                          Modifications are applied ONLY on the ranks that
//                          own the corresponding global wall — interior
//                          ranks' boundary rows connect to a neighbor via
//                          PaScaL_TDMA's alltoall and must remain intact.

class BoundaryApplier {
public:
  explicit BoundaryApplier(const Problem& problem) : problem_(problem) {}

  // Fills the ghost layer of `phi` on global boundary faces only, using the
  // per-field FieldBoundary `fbc`.  Interior halos remain untouched (they are
  // handled by MPI halo exchange).
  //
  //   t : current simulation time (passed to FaceBc value functions).
  void apply_ghost(field::ScalarField& phi,
                   const FieldBoundary& fbc,
                   real_t t = 0.0) const;

  // Modifies the cell-centered TDMA bands so that the BC on the wall axis is
  // enforced.  Only rows 0 (lower wall) and n_row-1 (upper wall) are touched,
  // and only on ranks that hold the corresponding global wall.
  //
  //   wall_axis : the direction along which TDMA is being solved.
  //   fbc       : per-field BC descriptor (e.g. problem.T for the thermal solver).
  //   axis_comm : the 1-D Cartesian sub-comm for `wall_axis`; its `west_rank`/
  //               `east_rank == MPI_PROC_NULL` identifies a global boundary
  //               rank.
  //   A,B,C,D   : PaScaL_TDMA bands, row-major [n_row × n_sys].
  void modify_tdma_row(Direction wall_axis,
                       const FieldBoundary& fbc,
                       const parallel::mpi::CartComm1D& axis_comm,
                       real_t* A, real_t* B, real_t* C, real_t* D,
                       int n_sys, int n_row) const;

  const Problem& problem() const { return problem_; }

private:
  const Problem& problem_;

  // Per-face implementations.
  void fill_face_periodic_  (field::ScalarField& phi, Direction d, Side s, const FaceBc& bc, real_t t) const;
  void fill_face_dirichlet_ (field::ScalarField& phi, Direction d, Side s, const FaceBc& bc, real_t t) const;
  void fill_face_neumann_   (field::ScalarField& phi, Direction d, Side s, const FaceBc& bc, real_t t) const;
};

} // namespace mpmstd::boundary
