#include "parallel/mpi/subdomain.hpp"

#include <stdexcept>

namespace mpmstd::parallel::mpi {

// ===== ParaRange =====================================================
// Distributes n_global indices [0, n_global-1] across nprocs ranks in
// contiguous blocks. Earlier ranks receive the larger blocks if the
// distribution is not exact.
//
// This matches PaScaL_TDMA_C/src/para_range.cpp behaviour, so subdomain
// extents line up with PaScaL_TDMA's plan creation.

ParaRange compute_para_range(int n_global, int nprocs, int rank) {
  if (nprocs <= 0 || rank < 0 || rank >= nprocs) {
    throw std::invalid_argument("compute_para_range: invalid nprocs/rank");
  }
  const int base      = n_global / nprocs;
  const int remainder = n_global % nprocs;

  ParaRange r;
  r.start = rank * base + (rank < remainder ? rank : remainder);
  const int count = base + (rank < remainder ? 1 : 0);
  r.end = r.start + count - 1;
  return r;
}

// ===== Subdomain =====================================================

Subdomain::Subdomain(const MpiTopology& topo, std::array<int, 3> n_global)
  : topo_(topo), n_global_(n_global) {

  for (int a = 0; a < 3; ++a) {
    const auto& axis = topo_.axis(static_cast<Direction>(a));
    auto r = compute_para_range(n_global_[a], axis.nprocs, axis.rank);
    n_interior_[a] = r.count();
    offset_[a]     = r.start;
    n_total_[a]    = n_interior_[a] + 2 * kHaloWidth;
  }

  build_datatypes_();
}

Subdomain::~Subdomain() {
  for (int a = 0; a < 3; ++a) {
    for (int s = 0; s < 2; ++s) {
      if (halo_type_[a][s] != MPI_DATATYPE_NULL) {
        MPI_Type_free(&halo_type_[a][s]);
      }
    }
  }
}

void Subdomain::build_datatypes_() {
  // Array shape (row-major): (n_total[0], n_total[1], n_total[2])
  // Strides:  k -> 1, j -> n_total[2], i -> n_total[1]*n_total[2]
  const int n0 = n_total_[0];
  const int n1 = n_total_[1];
  const int n2 = n_total_[2];

  // For each axis we create one datatype representing a single halo slab:
  // - axis 0 (X): a (1, n1, n2) slab = contiguous block of n1*n2 elements at a fixed i
  // - axis 1 (Y): a (n0, 1, n2) slab = strided
  // - axis 2 (Z): a (n0, n1, 1) slab = strided (most expensive)
  //
  // We use MPI_Type_create_subarray for clarity. One datatype per (axis, side),
  // though for plain halo exchange the type is the same shape on both sides;
  // we still create two so we can later swap to differing layouts (e.g. with
  // staggered velocity halos) without restructuring code.

  for (int axis = 0; axis < 3; ++axis) {
    int sizes   [3] = { n0, n1, n2 };
    int subsizes[3] = { n0, n1, n2 };
    subsizes[axis]  = 1;                  // one slab thick

    // Send/recv slab positions (kHaloWidth = 1)
    // - send-to-minus  : first interior plane along axis (index = kHaloWidth)
    // - recv-from-plus : top halo plane             (index = n_total - kHaloWidth)
    // - send-to-plus   : last interior plane        (index = n_total - 2*kHaloWidth)
    // - recv-from-minus: bottom halo plane          (index = 0)
    int idx_send_to_minus  = kHaloWidth;
    int idx_recv_from_plus = n_total_[axis] - kHaloWidth;
    int idx_send_to_plus   = n_total_[axis] - 2 * kHaloWidth;
    int idx_recv_from_minus = 0;

    // Linear offsets (in elements).
    auto plane_offset = [&](int axis_index, int slab_pos) {
      int starts[3] = { 0, 0, 0 };
      starts[axis_index] = slab_pos;
      return (starts[0] * n1 + starts[1]) * n2 + starts[2];
    };

    slab_offset_[axis][0] = plane_offset(axis, idx_send_to_minus);
    slab_offset_[axis][1] = plane_offset(axis, idx_recv_from_plus);
    slab_offset_[axis][2] = plane_offset(axis, idx_send_to_plus);
    slab_offset_[axis][3] = plane_offset(axis, idx_recv_from_minus);

    // The subsizes/sizes give the shape; starts are zero (we offset the
    // pointer at the MPI call site instead, for simplicity).
    int starts_zero[3] = {0, 0, 0};

    MPI_Type_create_subarray(3, sizes, subsizes, starts_zero,
                              MPI_ORDER_C,
                              sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                              &halo_type_[axis][0]);
    MPI_Type_commit(&halo_type_[axis][0]);

    // Same datatype reused for the "plus" side. Created separately so we can
    // free both symmetrically (and to leave room for future layout changes).
    MPI_Type_create_subarray(3, sizes, subsizes, starts_zero,
                              MPI_ORDER_C,
                              sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT,
                              &halo_type_[axis][1]);
    MPI_Type_commit(&halo_type_[axis][1]);
  }
}

void Subdomain::exchange_halo(real_t* data) const {
  // Six MPI_Sendrecv calls (two per axis). We exchange axis by axis.
  // After exchanging axis X, halos along X are filled including the
  // corner halos when the buffer was also receiving from Y/Z neighbors
  // in previous iterations; however, for the canonical structured-FDM
  // case we exchange axis-by-axis, and corner halos (only needed for
  // cross-derivative stencils) are typically left untouched here.
  //
  // For now we exchange faces only (no edges/corners). If a stencil
  // later needs corners, we extend this routine.

  for (int axis = 0; axis < 3; ++axis) {
    const auto& a = topo_.axis(static_cast<Direction>(axis));

    // Send to minus, receive from plus
    real_t* send_to_minus  = data + slab_offset_[axis][0];
    real_t* recv_from_plus = data + slab_offset_[axis][1];
    MPI_Sendrecv(send_to_minus,  1, halo_type_[axis][0], a.west_rank, /*tag=*/100 + axis,
                  recv_from_plus, 1, halo_type_[axis][1], a.east_rank, /*tag=*/100 + axis,
                  a.comm, MPI_STATUS_IGNORE);

    // Send to plus, receive from minus
    real_t* send_to_plus    = data + slab_offset_[axis][2];
    real_t* recv_from_minus = data + slab_offset_[axis][3];
    MPI_Sendrecv(send_to_plus,    1, halo_type_[axis][1], a.east_rank, /*tag=*/200 + axis,
                  recv_from_minus, 1, halo_type_[axis][0], a.west_rank, /*tag=*/200 + axis,
                  a.comm, MPI_STATUS_IGNORE);
  }
}

} // namespace mpmstd::parallel::mpi
