#pragma once

#include <mpi.h>
#include <string>

namespace mpmstd::parallel::mpi {

// RAII wrapper around MPI_Init / MPI_Finalize. Exactly one instance per process.
//
// Also exposes a node-local communicator (split by MPI_COMM_TYPE_SHARED), used
// to bind one MPI rank per GPU in CUDA builds.

class MpiContext {
public:
  MpiContext(int* argc, char*** argv);
  ~MpiContext();

  MpiContext(const MpiContext&) = delete;
  MpiContext& operator=(const MpiContext&) = delete;

  int      world_rank() const { return world_rank_; }
  int      world_size() const { return world_size_; }
  MPI_Comm world_comm() const { return MPI_COMM_WORLD; }

  int      node_rank() const { return node_rank_; }
  int      node_size() const { return node_size_; }
  MPI_Comm node_comm() const { return node_comm_; }

  // Convenience: true if this is the lead rank of the world.
  bool is_root() const { return world_rank_ == 0; }

  // Hostname for diagnostics.
  const std::string& hostname() const { return hostname_; }

private:
  int      world_rank_ = -1;
  int      world_size_ = 0;
  int      node_rank_  = -1;
  int      node_size_  = 0;
  MPI_Comm node_comm_  = MPI_COMM_NULL;
  std::string hostname_;
};

} // namespace mpmstd::parallel::mpi
