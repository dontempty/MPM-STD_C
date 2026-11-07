#include "transpose_transport.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace fft_test {

const char* transpose_mode_name() {
  static const char* const name = [] {
    const char* value = std::getenv("FFT_TEST_TRANSPOSE");
    if (value == nullptr || std::strcmp(value, "alltoallv") == 0) return "alltoallv";
    if (std::strcmp(value, "pairwise") == 0) return "pairwise";
    if (std::strcmp(value, "nccl") == 0) return "nccl";
    throw std::invalid_argument(std::string("FFT_TEST_TRANSPOSE must be alltoallv, pairwise or nccl, got: ") + value);
  }();
  return name;
}

const char* tdma_mode_name() {
  static const char* const name = [] {
    const char* value = std::getenv("FFT_TEST_TDMA_A2A");
    if (value == nullptr || std::strcmp(value, "alltoallv") == 0) return "alltoallv";
    if (std::strcmp(value, "pairwise") == 0) return "pairwise";
    throw std::invalid_argument(std::string("FFT_TEST_TDMA_A2A must be alltoallv or pairwise, got: ") + value);
  }();
  return name;
}

#ifdef MPMSTD_GPU

namespace {

struct Entry {
  MPI_Comm comm = MPI_COMM_NULL;
  const char* mode = "alltoallv";
  ncclComm_t nccl = nullptr;
};

std::vector<Entry> entries;

const Entry* find_entry(MPI_Comm comm) {
  for (const Entry& e : entries)
    if (e.comm == comm) return &e;
  return nullptr;
}

int pairwise(const double* send, const int* send_counts, const int* send_displs, double* recv, const int* recv_counts,
             const int* recv_displs, MPI_Comm comm) {
  int me = 0, size = 1;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &size);
  if (send_counts[me] > 0 || recv_counts[me] > 0) {   // 디바이스 버퍼라 여기서 복사할 수 없다. 조용히 빠뜨리면 오답이다.
    std::fprintf(stderr, "FFT_test pairwise: self block was not removed (MPMSTD_A2A_SELF must stay copy)\n");
    return MPI_ERR_OTHER;
  }
  for (int step = 1; step < size; ++step) {
    const int destination = (me + step) % size, source = (me - step + size) % size;
    MPI_Request requests[2];
    int n = 0, status = MPI_SUCCESS;
    if (recv_counts[source] > 0)
      status = MPI_Irecv(recv + recv_displs[source], recv_counts[source], MPI_DOUBLE, source, 0, comm, &requests[n++]);
    if (status == MPI_SUCCESS && send_counts[destination] > 0)
      status = MPI_Isend(send + send_displs[destination], send_counts[destination], MPI_DOUBLE, destination, 0, comm, &requests[n++]);
    if (status == MPI_SUCCESS && n > 0) status = MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
    if (status != MPI_SUCCESS) return status;
  }
  return MPI_SUCCESS;
}

int report_nccl(const ncclResult_t result, const char* what) {
  std::fprintf(stderr, "FFT_test NCCL failed in %s: %s\n", what, ncclGetErrorString(result));
  std::fflush(stderr);
  return MPI_ERR_OTHER;
}

int through_nccl(const double* send, const int* send_counts, const int* send_displs, double* recv, const int* recv_counts,
                 const int* recv_displs, MPI_Comm comm, const ncclComm_t nccl) {
  int me = 0, size = 1;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &size);
  bool any = false;
  for (int r = 0; r < size; ++r)
    if (r != me && (send_counts[r] > 0 || recv_counts[r] > 0)) any = true;
  if (!any) return MPI_SUCCESS;   // x 를 나누지 않으면 블록 → x 라인 전치는 자기 몫뿐이다
  ncclResult_t result = ncclGroupStart();
  if (result != ncclSuccess) return report_nccl(result, "ncclGroupStart");
  for (int step = 1; step < size; ++step) {
    const int destination = (me + step) % size, source = (me - step + size) % size;
    if (send_counts[destination] > 0) {
      result = ncclSend(send + send_displs[destination], static_cast<std::size_t>(send_counts[destination]), ncclDouble, destination, nccl, nullptr);
      if (result != ncclSuccess) return report_nccl(result, "ncclSend");
    }
    if (recv_counts[source] > 0) {
      result = ncclRecv(recv + recv_displs[source], static_cast<std::size_t>(recv_counts[source]), ncclDouble, source, nccl, nullptr);
      if (result != ncclSuccess) return report_nccl(result, "ncclRecv");
    }
  }
  result = ncclGroupEnd();
  if (result != ncclSuccess) return report_nccl(result, "ncclGroupEnd");
  // MPI_Alltoallv 와 같은 의미: 돌아올 때 전송이 끝나 있어야 한다 (호출자가 바로 언팩하고 시간을 잰다)
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "FFT_test NCCL: cudaDeviceSynchronize failed\n");
    return MPI_ERR_OTHER;
  }
  return MPI_SUCCESS;
}

}  // namespace

void register_transpose_comm(MPI_Comm comm) {
  Entry entry;
  entry.comm = comm;
  entry.mode = transpose_mode_name();
  if (std::strcmp(entry.mode, "nccl") == 0) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    ncclUniqueId id;
    if (rank == 0 && ncclGetUniqueId(&id) != ncclSuccess) throw std::runtime_error("ncclGetUniqueId failed");
    PMPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    const ncclResult_t result = ncclCommInitRank(&entry.nccl, size, id, rank);
    if (result != ncclSuccess) throw std::runtime_error(std::string("ncclCommInitRank failed: ") + ncclGetErrorString(result));
  }
  entries.push_back(entry);
}

void register_tdma_comm(MPI_Comm comm) {
  Entry entry;
  entry.comm = comm;
  entry.mode = tdma_mode_name();
  entries.push_back(entry);
}

void release_transpose_comms() {
  for (Entry& e : entries)
    if (e.nccl) ncclCommDestroy(e.nccl);
  entries.clear();
}

}  // namespace fft_test

// PMPI 가로채기. 실행파일의 정의가 libmpi 보다 먼저 잡히므로 src/ 의 호출도 여기로 온다.
extern "C" int MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void* recvbuf,
                             const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
  using namespace fft_test;
  const Entry* entry = find_entry(comm);
  if (entry == nullptr || std::strcmp(entry->mode, "alltoallv") == 0 || sendtype != MPI_DOUBLE || recvtype != MPI_DOUBLE)
    return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
  const double* send = static_cast<const double*>(sendbuf);
  double* recv = static_cast<double*>(recvbuf);
  if (std::strcmp(entry->mode, "pairwise") == 0) return pairwise(send, sendcounts, sdispls, recv, recvcounts, rdispls, comm);
  return through_nccl(send, sendcounts, sdispls, recv, recvcounts, rdispls, comm, entry->nccl);
}

#else   // CPU 빌드: 가로채지 않는다

namespace fft_test {
void register_transpose_comm(MPI_Comm) {}
void register_tdma_comm(MPI_Comm) {}
void release_transpose_comms() {}
}  // namespace fft_test

#endif
