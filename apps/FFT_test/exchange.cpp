#include "exchange.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "device_loop.hpp"

#ifdef MPMSTD_GPU
#include "common/cuda_check.hpp"
#endif

namespace fft_test {

bool host_staging_requested() {
  const char* mode = std::getenv("FFT_TEST_MPI_MODE");
  return mode != nullptr && std::strcmp(mode, "host") == 0;
}

const char* a2a_mode_name() {
  static const char* const name = [] {
    const char* value = std::getenv("FFT_TEST_A2A");
    if (value == nullptr || std::strcmp(value, "alltoallv") == 0) return "alltoallv";
    if (std::strcmp(value, "p2p") == 0) return "p2p";
    if (std::strcmp(value, "pairwise") == 0) return "pairwise";
    throw std::invalid_argument(std::string("FFT_TEST_A2A must be alltoallv, p2p or pairwise, got: ") + value);
  }();
  return name;
}

void alltoallv_backend(const real_t* send, const std::vector<int>& send_counts,
                       const std::vector<int>& send_displs, real_t* recv,
                       const std::vector<int>& recv_counts, const std::vector<int>& recv_displs,
                       MPI_Comm comm, const bool host_staging) {
  int status = MPI_SUCCESS;
#ifdef MPMSTD_GPU
  synchronize_device();
  if (host_staging) {
    // 호스트 경유는 대조용. 카운트 합 = 마지막 변위 + 마지막 개수.
    const std::size_t n_send = static_cast<std::size_t>(send_displs.back()) + send_counts.back();
    const std::size_t n_recv = static_cast<std::size_t>(recv_displs.back()) + recv_counts.back();
    std::vector<real_t> send_host(n_send), recv_host(n_recv);
    MPMSTD_CUDA_CHECK(cudaMemcpy(send_host.data(), send, n_send * sizeof(real_t), cudaMemcpyDeviceToHost));
    status = MPI_Alltoallv(send_host.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                           recv_host.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE, comm);
    MPMSTD_CUDA_CHECK(cudaMemcpy(recv, recv_host.data(), n_recv * sizeof(real_t), cudaMemcpyHostToDevice));
  } else {
    int me = 0;
    MPI_Comm_rank(comm, &me);
    if (send_counts[me] != recv_counts[me])
      throw std::runtime_error("FFT_test alltoallv: self send/recv counts differ");
    if (send_counts[me] > 0)
      MPMSTD_CUDA_CHECK(cudaMemcpyAsync(recv + recv_displs[me], send + send_displs[me],
                                        static_cast<std::size_t>(send_counts[me]) * sizeof(real_t),
                                        cudaMemcpyDeviceToDevice));
    const char* mode = a2a_mode_name();
    if (std::strcmp(mode, "alltoallv") == 0) {
      std::vector<int> peer_send(send_counts), peer_recv(recv_counts);
      peer_send[me] = 0;
      peer_recv[me] = 0;
      status = MPI_Alltoallv(send, peer_send.data(), send_displs.data(), MPI_DOUBLE, recv, peer_recv.data(),
                             recv_displs.data(), MPI_DOUBLE, comm);
    } else {
      int size = 1;
      MPI_Comm_size(comm, &size);
      if (std::strcmp(mode, "p2p") == 0) {
        std::vector<MPI_Request> requests;
        requests.reserve(2 * static_cast<std::size_t>(size));
        for (int step = 1; step < size && status == MPI_SUCCESS; ++step) {
          const int destination = (me + step) % size, source = (me - step + size) % size;
          if (send_counts[destination] > 0) {
            requests.emplace_back();
            status = MPI_Isend(send + send_displs[destination], send_counts[destination], MPI_DOUBLE, destination, 0, comm,
                               &requests.back());
          }
          if (status == MPI_SUCCESS && recv_counts[source] > 0) {
            requests.emplace_back();
            status = MPI_Irecv(recv + recv_displs[source], recv_counts[source], MPI_DOUBLE, source, 0, comm, &requests.back());
          }
        }
        if (status == MPI_SUCCESS)
          status = MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
      } else {   // pairwise
        for (int step = 1; step < size && status == MPI_SUCCESS; ++step) {
          const int destination = (me + step) % size, source = (me - step + size) % size;
          status = MPI_Sendrecv(send + send_displs[destination], send_counts[destination], MPI_DOUBLE, destination, 0,
                                recv + recv_displs[source], recv_counts[source], MPI_DOUBLE, source, 0, comm,
                                MPI_STATUS_IGNORE);
        }
      }
    }
    synchronize_device();   // self 복사 완료
  }
#else
  (void)host_staging;
  status = MPI_Alltoallv(send, send_counts.data(), send_displs.data(), MPI_DOUBLE, recv, recv_counts.data(),
                         recv_displs.data(), MPI_DOUBLE, comm);
#endif
  if (status != MPI_SUCCESS) throw std::runtime_error("FFT_test: MPI_Alltoallv failed");
}

}  // namespace fft_test
