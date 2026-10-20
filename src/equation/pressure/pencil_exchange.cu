#include "equation/pressure/pencil_transpose.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"

// GPU 백엔드 — PaScaL_TDMA(pascal_tdma_cuda.cu, alltoallv_double)와 같은 구조:
//   팩 커널(디바이스) → 디바이스 송신 버퍼를 **그대로 MPI_Alltoallv** (CUDA-aware MPI, GPU↔GPU)
//   → 디바이스 수신 버퍼 → 언팩 커널. host_staging(PASCAL_TDMA_MPI_MODE=host)일 때만 호스트 경유.

namespace mpmstd::equation::pressure {

namespace {

enum : int { kThreads = 256 };

__global__ void gather_kernel(real_t* send, const real_t* src, const int* index, const long long n,
                              const int width, const real_t scale, const real_t subtract) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= n * width) return;
  const long long e = q / width;
  const int c = static_cast<int>(q - e * width);
  send[q] = src[static_cast<long long>(index[e]) * width + c] * scale - subtract;
}

__global__ void scatter_kernel(real_t* dst, const real_t* recv, const int* index, const int* mirror,
                               const long long n, const int width) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= n * width) return;
  const long long e = q / width;
  const int c = static_cast<int>(q - e * width);
  dst[static_cast<long long>(index[e]) * width + c] = recv[q];
  if (mirror) dst[static_cast<long long>(mirror[e]) * width + c] = recv[q];
}

int* upload_indices(const std::vector<int>& host) {
  if (host.empty()) return nullptr;
  int* device = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(int)));
  MPMSTD_CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
  return device;
}

int blocks_for(const long long n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

}  // namespace

void prepare_exchange(PencilExchange& e) {
  e.device_send_index = upload_indices(e.send_index);
  e.device_recv_index = upload_indices(e.recv_index);
  e.device_recv_mirror = upload_indices(e.recv_mirror);
  // 크기 0 버퍼도 유효해야 하므로 최소 1 원소
  e.send_buffer.assign(std::max<std::size_t>(1, e.send_index.size() * e.width));
  e.recv_buffer.assign(std::max<std::size_t>(1, e.recv_index.size() * e.width));
}

void release_exchange(PencilExchange& e) {
  for (int** p : {&e.device_send_index, &e.device_recv_index, &e.device_recv_mirror}) {
    if (*p) cudaFree(*p);
    *p = nullptr;
  }
}

void run_exchange(PencilExchange& e, const real_t* src, real_t* dst, MPI_Comm comm,
                  const bool host_staging, const real_t scale, const real_t subtract) {
  const long long n_send = static_cast<long long>(e.send_index.size());
  const long long n_recv = static_cast<long long>(e.recv_index.size());
  real_t* send = e.send_buffer.data();
  real_t* recv = e.recv_buffer.data();
  if (n_send > 0) {
    gather_kernel<<<blocks_for(n_send * e.width), kThreads>>>(send, src, e.device_send_index, n_send,
                                                              e.width, scale, subtract);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  // 디바이스 버퍼를 MPI 에 넘기기 직전 — 팩 커널 완료를 보장 (PaScaL 의 cudaStreamSynchronize 와 같은 자리)
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
  int status = MPI_SUCCESS;
  if (!host_staging) {
    status = MPI_Alltoallv(send, e.send_counts.data(), e.send_displs.data(), MPI_DOUBLE, recv,
                           e.recv_counts.data(), e.recv_displs.data(), MPI_DOUBLE, comm);
  } else {
    std::vector<real_t> send_host(e.send_buffer.size()), recv_host(e.recv_buffer.size());
    MPMSTD_CUDA_CHECK(cudaMemcpy(send_host.data(), send, send_host.size() * sizeof(real_t),
                                 cudaMemcpyDeviceToHost));
    status = MPI_Alltoallv(send_host.data(), e.send_counts.data(), e.send_displs.data(), MPI_DOUBLE,
                           recv_host.data(), e.recv_counts.data(), e.recv_displs.data(), MPI_DOUBLE,
                           comm);
    MPMSTD_CUDA_CHECK(cudaMemcpy(recv, recv_host.data(), recv_host.size() * sizeof(real_t),
                                 cudaMemcpyHostToDevice));
  }
  if (status != MPI_SUCCESS) throw std::runtime_error("pressure pencil: MPI_Alltoallv failed");
  if (n_recv > 0) {
    scatter_kernel<<<blocks_for(n_recv * e.width), kThreads>>>(dst, recv, e.device_recv_index,
                                                               e.device_recv_mirror, n_recv, e.width);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace mpmstd::equation::pressure
