#include "parallel/mpi/mpi_context.hpp"

#include <stdexcept>
#include <vector>

namespace mpmstd::parallel::mpi {

MpiContext::MpiContext(int* argc, char*** argv) {
  int already_initialized = 0;
  MPI_Initialized(&already_initialized);
  if (!already_initialized) {
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, &provided);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  // Node-local communicator (one comm per physical node).
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                      world_rank_, MPI_INFO_NULL, &node_comm_);
  MPI_Comm_rank(node_comm_, &node_rank_);
  MPI_Comm_size(node_comm_, &node_size_);

  // Hostname for logging.
  std::vector<char> name(MPI_MAX_PROCESSOR_NAME, '\0');
  int len = 0;
  MPI_Get_processor_name(name.data(), &len);
  hostname_.assign(name.data(), static_cast<std::size_t>(len));
}

MpiContext::~MpiContext() {
  if (node_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&node_comm_);
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}

} // namespace mpmstd::parallel::mpi
