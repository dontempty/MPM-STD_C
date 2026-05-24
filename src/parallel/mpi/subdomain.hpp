#pragma once

#include "common/macros.hpp"
#include "common/types.hpp"
#include "common/direction.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <mpi.h>
#include <array>

namespace mpmstd::parallel::mpi {

// Computes the (start, end) interior index range for a given (global N, nprocs, rank).
// Inclusive 0-based interior range; total local interior count = end - start + 1.
struct ParaRange {
  int start;     // first global interior index belonging to this rank (0-based)
  int end;       // last  global interior index belonging to this rank (0-based, inclusive)
  int count() const { return end - start + 1; }
};

ParaRange compute_para_range(int n_global, int nprocs, int rank);

// Subdomain decomposition for one MPI rank.
//
// Conventions
// -----------
//   n_global[d]  : global interior cell count along axis d (n1m, n2m, n3m)
//   n_interior[d]: local interior cell count (this rank's share of n_global)
//   n_total[d]   : n_interior[d] + 2*kHaloWidth, total stride along axis d.
//
// Arrays of shape (n_total[0], n_total[1], n_total[2]) are addressed with
//   idx = i * stride_x + j * stride_y + k * stride_z
// where stride_z = 1, stride_y = n_total[2], stride_x = n_total[1] * n_total[2]
// (i.e. row-major, with the slowest dimension being x).
//
// Halo cells occupy indices 0 and n_total[d]-1 along each axis.
//
// Each Subdomain owns the derived-datatype set needed for ghost-cell exchange
// along each axis; the exchange routine is defined for `real_t` arrays.

class Subdomain {
public:
  Subdomain(const MpiTopology& topo,
            std::array<int, 3> n_global);
  ~Subdomain();

  Subdomain(const Subdomain&) = delete;
  Subdomain& operator=(const Subdomain&) = delete;

  const MpiTopology& topology() const { return topo_; }

  // Global extents.
  std::array<int, 3> n_global()  const { return n_global_; }

  // Local extents (no halos).
  std::array<int, 3> n_interior() const { return n_interior_; }
  int n_interior(Direction d) const { return n_interior_[to_int(d)]; }

  // Local strides including halos.
  std::array<int, 3> n_total() const { return n_total_; }
  int n_total(Direction d) const { return n_total_[to_int(d)]; }

  // Global offset of the first local interior cell along axis d.
  std::array<int, 3> global_offset() const { return offset_; }
  int global_offset(Direction d) const { return offset_[to_int(d)]; }

  // Total element count for a 3D array with halos.
  std::size_t n_elements() const {
    return static_cast<std::size_t>(n_total_[0]) * n_total_[1] * n_total_[2];
  }

  // Row-major linear index. i,j,k are 0-based, including halo positions.
  int linear_index(int i, int j, int k) const {
    return (i * n_total_[1] + j) * n_total_[2] + k;
  }

  // Exchange halos for a 3D real-valued array allocated of shape n_total().
  // Caller is responsible for the buffer; this routine fills the halo planes
  // from neighboring ranks (or applies periodic wrap-around when the topology
  // is periodic on that axis).
  //
  // On CPU build the pointer is a host pointer. On CUDA build with CUDA-aware
  // MPI, the same call works with a device pointer (cuda_aware_mpi.hpp wraps
  // the choice).
  void exchange_halo(real_t* data) const;

private:
  const MpiTopology& topo_;
  std::array<int, 3> n_global_{};
  std::array<int, 3> n_interior_{};
  std::array<int, 3> n_total_{};
  std::array<int, 3> offset_{};

  // Per-axis derived datatypes: send/recv buffers of one full halo plane.
  // halo_type_[axis][side]: contiguous block of one halo slab in the row-major
  // array.
  // side 0 = minus (lower) face, 1 = plus (upper) face.
  MPI_Datatype halo_type_[3][2] = {{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL},
                                    {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL},
                                    {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL}};

  // Offsets (in elements) of the slabs participating in the exchange:
  // [axis][slab_role] where slab_role:
  //   0: send-from-interior-to-minus-neighbor (= first interior plane)
  //   1: recv-into-plus-halo                  (= top halo plane)
  //   2: send-from-interior-to-plus-neighbor  (= last interior plane)
  //   3: recv-into-minus-halo                 (= bottom halo plane)
  int slab_offset_[3][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0}};

  void build_datatypes_();
};

} // namespace mpmstd::parallel::mpi
