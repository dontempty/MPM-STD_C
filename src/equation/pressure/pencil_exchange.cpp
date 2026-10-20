#include "equation/pressure/pencil_transpose.hpp"

#include <cstddef>
#include <stdexcept>

// CPU 백엔드 — 인덱스 표는 호스트 벡터를 그대로 쓰고, 팩·언팩은 호스트 루프 (GPU 쌍둥이 pencil_exchange.cu).

namespace mpmstd::equation::pressure {

void prepare_exchange(PencilExchange& e) {
  e.device_send_index = e.send_index.data();
  e.device_recv_index = e.recv_index.data();
  e.device_recv_mirror = e.recv_mirror.empty() ? nullptr : e.recv_mirror.data();
  e.send_buffer.assign(e.send_index.size() * e.width);
  e.recv_buffer.assign(e.recv_index.size() * e.width);
}

void release_exchange(PencilExchange& e) {
  e.device_send_index = e.device_recv_index = e.device_recv_mirror = nullptr;
}

void run_exchange(PencilExchange& e, const real_t* src, real_t* dst, MPI_Comm comm,
                  const bool host_staging, const real_t scale, const real_t subtract) {
  (void)host_staging;  // CPU 는 원래 호스트 메모리
  const int w = e.width;
  real_t* send = e.send_buffer.data();
  real_t* recv = e.recv_buffer.data();
  const std::size_t n_send = e.send_index.size(), n_recv = e.recv_index.size();
  for (std::size_t n = 0; n < n_send; ++n)
    for (int c = 0; c < w; ++c)
      send[n * w + c] = src[static_cast<std::size_t>(e.send_index[n]) * w + c] * scale - subtract;
  if (MPI_Alltoallv(send, e.send_counts.data(), e.send_displs.data(), MPI_DOUBLE, recv,
                    e.recv_counts.data(), e.recv_displs.data(), MPI_DOUBLE, comm) != MPI_SUCCESS)
    throw std::runtime_error("pressure pencil: MPI_Alltoallv failed");
  for (std::size_t n = 0; n < n_recv; ++n)
    for (int c = 0; c < w; ++c) {
      dst[static_cast<std::size_t>(e.recv_index[n]) * w + c] = recv[n * w + c];
      if (e.device_recv_mirror) dst[static_cast<std::size_t>(e.recv_mirror[n]) * w + c] = recv[n * w + c];
    }
}

}  // namespace mpmstd::equation::pressure
