#include "parallel/topology/topology.hpp"

#include <stdexcept>
#include <string>

#include "parallel/mpi_context.hpp"

namespace mpmstd::parallel {

namespace {

// Cartesian 통신자에서 keep_dim 축만 남긴 1D 서브커뮤니케이터를 만든다.
// 서브커뮤니케이터가 부모의 Cartesian 구조(주기성 포함)를 물려받으므로,
// MPI_Cart_shift 가 주는 이웃 랭크는 a.comm 안에서 그대로 쓸 수 있고
// 비주기 축의 바깥쪽은 MPI_PROC_NULL 이 된다.
AxisComm make_axis_comm(MPI_Comm cart_comm, int keep_dim) {
  AxisComm a;
  int remain[3] = {0, 0, 0};
  remain[keep_dim] = 1;
  MPI_Cart_sub(cart_comm, remain, &a.comm);
  MPI_Comm_rank(a.comm, &a.rank);
  MPI_Comm_size(a.comm, &a.nprocs);
  MPI_Cart_shift(a.comm, /*direction=*/0, /*displacement=*/1,
                 &a.minus_neighbor, &a.plus_neighbor);
  return a;
}

}  // namespace

Topology::Topology(const MpiContext& mpi, int npx, int npy, int npz,
                   bool periodic_x, bool periodic_y, bool periodic_z) {
  if (npx <= 0 || npy <= 0 || npz <= 0)
    throw std::invalid_argument("Topology: process counts must be > 0, got "
                                + std::to_string(npx) + "x" + std::to_string(npy)
                                + "x" + std::to_string(npz));
  if (npx * npy * npz != mpi.world_size())
    throw std::runtime_error("Topology: npx*npy*npz = " + std::to_string(npx * npy * npz)
                             + " does not match MPI rank count "
                             + std::to_string(mpi.world_size()));

  int dims[3] = {npx, npy, npz};
  int periods[3] = {periodic_x ? 1 : 0, periodic_y ? 1 : 0, periodic_z ? 1 : 0};
  MPI_Cart_create(mpi.world_comm(), 3, dims, periods, /*reorder=*/0, &cart_comm_);

  x_ = make_axis_comm(cart_comm_, 0);
  y_ = make_axis_comm(cart_comm_, 1);
  z_ = make_axis_comm(cart_comm_, 2);
  x_.periodic = periodic_x;
  y_.periodic = periodic_y;
  z_.periodic = periodic_z;
}

Topology::~Topology() {
  if (z_.comm != MPI_COMM_NULL) MPI_Comm_free(&z_.comm);
  if (y_.comm != MPI_COMM_NULL) MPI_Comm_free(&y_.comm);
  if (x_.comm != MPI_COMM_NULL) MPI_Comm_free(&x_.comm);
  if (cart_comm_ != MPI_COMM_NULL) MPI_Comm_free(&cart_comm_);
}

}  // namespace mpmstd::parallel
