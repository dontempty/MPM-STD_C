#include "parallel/mpi/mpi_topology.hpp"

#include <stdexcept>
#include <string>

namespace mpmstd::parallel::mpi {

MpiTopology::MpiTopology(const MpiContext& ctx,
                          std::array<int, 3>  dims,
                          std::array<bool, 3> periodic)
  : dims_(dims), periodic_(periodic) {

  const int requested = dims_[0] * dims_[1] * dims_[2];
  if (requested != ctx.world_size()) {
    throw std::runtime_error(
      "MpiTopology: dims product (" + std::to_string(requested) +
      ") does not match MPI world size (" + std::to_string(ctx.world_size()) + ")");
  }

  int periods_int[3] = {
    periodic_[0] ? 1 : 0,
    periodic_[1] ? 1 : 0,
    periodic_[2] ? 1 : 0,
  };
  int dims_int[3] = { dims_[0], dims_[1], dims_[2] };

  // reorder = 0 keeps rank mapping consistent across runs.
  MPI_Cart_create(ctx.world_comm(), 3, dims_int, periods_int,
                  /*reorder=*/0, &cart_comm_);

  int coords_int[3] = {0, 0, 0};
  int my_rank = -1;
  MPI_Comm_rank(cart_comm_, &my_rank);
  MPI_Cart_coords(cart_comm_, my_rank, 3, coords_int);
  coords_ = { coords_int[0], coords_int[1], coords_int[2] };

  for (int a = 0; a < 3; ++a) {
    build_axis_comm_(a);
  }
}

MpiTopology::~MpiTopology() {
  for (auto& a : axis_) {
    if (a.comm != MPI_COMM_NULL) {
      MPI_Comm_free(&a.comm);
    }
  }
  if (cart_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm_);
  }
}

void MpiTopology::build_axis_comm_(int axis) {
  // remain_dims marks which Cartesian directions remain in the sub-comm.
  int remain_dims[3] = {0, 0, 0};
  remain_dims[axis]  = 1;

  CartComm1D& a = axis_[axis];
  MPI_Cart_sub(cart_comm_, remain_dims, &a.comm);
  MPI_Comm_rank(a.comm, &a.rank);
  MPI_Comm_size(a.comm, &a.nprocs);

  // Compute neighbors via MPI_Cart_shift on the AXIS sub-communicator (1D).
  // The sub-comm retains the Cartesian topology from the parent, so the
  // returned west/east ranks are valid *within a.comm* (which is the comm
  // we later pass to MPI_Sendrecv).
  // With periodic = false, the outer ranks get MPI_PROC_NULL — boundary
  // halo exchanges then become no-ops at walls.
  MPI_Cart_shift(a.comm, /*direction=*/0, /*disp=*/1, &a.west_rank, &a.east_rank);
}

} // namespace mpmstd::parallel::mpi
