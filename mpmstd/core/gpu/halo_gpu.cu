#include "core/halo.hpp"

#include <cuda_runtime.h>

// GPU implementations of the core communication free functions. Compiled ONLY
// in the GPU build (`make gpu`); the cpu-only build never sees the gpu/ folder,
// so it carries zero CUDA dependency (rev.2 §6).

namespace mpmstd::core {

void exchange_halo_gpu(GpuField& f, const Subdomain& sub) {
  // CUDA-aware MPI (rev.2 C2): hand the DEVICE pointer straight to the same
  // Subdomain::exchange_halo routine — no host staging. The underlying
  // MPI_Sendrecv operates device-to-device when the MPI is CUDA-aware.
  sub.exchange_halo(f.data());
}

void bind_gpu_to_local_rank_gpu(const MpiContext& ctx) {
  // rev.2 C2 — 1 MPI rank == 1 GPU. node_rank() is the node-local rank (split by
  // MPI_COMM_TYPE_SHARED), so this maps each rank to a distinct device and makes
  // the "many ranks per GPU" mistake structurally impossible.
  int ndev = 0;
  cudaError_t err = cudaGetDeviceCount(&ndev);
  if (err == cudaSuccess && ndev > 0) {
    cudaSetDevice(ctx.node_rank() % ndev);
  }
}

} // namespace mpmstd::core
