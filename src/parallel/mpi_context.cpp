#include "parallel/mpi_context.hpp"

namespace mpmstd::parallel {

MpiContext::MpiContext(int* argc, char*** argv) {
  int already_initialized = 0;
  MPI_Initialized(&already_initialized);
  if (!already_initialized) {
    MPI_Init(argc, argv);
    we_initialized_ = true;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  // 노드 로컬 커뮤니케이터 — key 로 world_rank 를 넘겨 노드 안의 랭크 순서가
  // world 순서와 일치하게 한다.
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank_,
                      MPI_INFO_NULL, &node_comm_);
  MPI_Comm_rank(node_comm_, &node_rank_);
  MPI_Comm_size(node_comm_, &node_size_);
}

MpiContext::~MpiContext() {
  if (node_comm_ != MPI_COMM_NULL) MPI_Comm_free(&node_comm_);
  if (we_initialized_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
  }
}

}  // namespace mpmstd::parallel
