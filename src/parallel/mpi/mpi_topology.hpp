#pragma once

#include "common/direction.hpp"
#include "parallel/mpi/mpi_context.hpp"

#include <mpi.h>
#include <array>

namespace mpmstd::parallel::mpi {

// 1D Cartesian sub-communicator info, mirrors the heat_gpu CartComm1D struct.
struct CartComm1D {
  MPI_Comm comm        = MPI_COMM_NULL;
  int      rank        = -1;
  int      nprocs      = 0;
  int      west_rank   = MPI_PROC_NULL;   // neighbor on the negative side
  int      east_rank   = MPI_PROC_NULL;   // neighbor on the positive side
};

// 3D Cartesian topology plus three 1D sub-communicators (one per axis).
// The 2D sub-communicators used for the pressure-FFT transpose (e.g.
// comm_x1n2 in PaScaL_TCS) are introduced later, when the pressure solver is
// implemented (M4). For M0..M3 we only need 3D + 3 axis comms.

class MpiTopology {
public:
  MpiTopology(const MpiContext& ctx,
              std::array<int, 3>  dims,
              std::array<bool, 3> periodic);
  ~MpiTopology();

  MpiTopology(const MpiTopology&) = delete;
  MpiTopology& operator=(const MpiTopology&) = delete;

  MPI_Comm cart_comm() const { return cart_comm_; }
  std::array<int, 3>  dims()     const { return dims_; }
  std::array<bool, 3> periodic() const { return periodic_; }
  std::array<int, 3>  coords()   const { return coords_; }

  const CartComm1D& axis(Direction d) const { return axis_[to_int(d)]; }

  // comm_xz: sub-communicator containing all ranks with the same Y-coordinate
  // (np1 * np3 ranks total). Used by the pressure-FFT pencil decomposition for
  // the distributed z-TDMA over all X–Z process pairs.
  MPI_Comm comm_xz() const { return comm_xz_; }
  int      rank_xz() const { return rank_xz_; }
  int      size_xz() const { return size_xz_; }

private:
  MPI_Comm                cart_comm_ = MPI_COMM_NULL;
  std::array<int, 3>      dims_{};
  std::array<bool, 3>     periodic_{};
  std::array<int, 3>      coords_{};
  std::array<CartComm1D, 3> axis_{};

  MPI_Comm comm_xz_ = MPI_COMM_NULL;
  int      rank_xz_ = 0;
  int      size_xz_ = 1;

  void build_axis_comm_(int axis);
};

} // namespace mpmstd::parallel::mpi
