#pragma once
// MPI_Init / MPI_Finalize 의 RAII 래퍼 — 프로세스당 하나, main 의 첫 객체.
// node_rank() 는 같은 물리 노드 안에서의 랭크 — 이후 GPU 단계에서
// "1 랭크 = 1 GPU" 바인딩(cudaSetDevice(node_rank % device_count))의 근거다.
//
// 우리가 초기화하지 않은 MPI 는 Finalize 하지 않는다 (we_initialized_).
// (MPM-STD_C 는 남이 초기화한 MPI 도 소멸자가 끝내 버리는 비대칭 소유권이 있었다)

#include <mpi.h>

namespace mpmstd::parallel {

class MpiContext {
 public:
  MpiContext(int* argc, char*** argv);
  ~MpiContext();
  MpiContext(const MpiContext&) = delete;
  MpiContext& operator=(const MpiContext&) = delete;

  MPI_Comm world_comm() const { return MPI_COMM_WORLD; }
  int world_rank() const { return world_rank_; }
  int world_size() const { return world_size_; }
  bool is_root() const { return world_rank_ == 0; }

  int node_rank() const { return node_rank_; }
  int node_size() const { return node_size_; }

 private:
  int world_rank_ = -1;
  int world_size_ = 0;
  int node_rank_ = -1;
  int node_size_ = 0;
  MPI_Comm node_comm_ = MPI_COMM_NULL;
  bool we_initialized_ = false;
};

}  // namespace mpmstd::parallel
