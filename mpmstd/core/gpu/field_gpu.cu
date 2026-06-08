#include "core/field_gpu.hpp"

#include <cuda_runtime.h>

// GpuField member definitions — device memory management. GPU build only.

namespace mpmstd::core {

GpuField::GpuField(const parallel::mpi::Subdomain& sub, std::string name)
  : name_(std::move(name)),
    n_total_(sub.n_total()),
    n_interior_(sub.n_interior()),
    offset_(sub.global_offset()),
    n_elements_(sub.n_elements()) {
  cudaMalloc(reinterpret_cast<void**>(&device_), n_elements_ * sizeof(real_t));
  cudaMemset(device_, 0, n_elements_ * sizeof(real_t));   // 0.0 bit pattern
}

GpuField::~GpuField() {
  if (device_) {
    cudaFree(device_);
    device_ = nullptr;
  }
}

void GpuField::to_device(const real_t* host_src) {
  cudaMemcpy(device_, host_src, n_elements_ * sizeof(real_t), cudaMemcpyHostToDevice);
}

void GpuField::to_host(real_t* host_dst) const {
  cudaMemcpy(host_dst, device_, n_elements_ * sizeof(real_t), cudaMemcpyDeviceToHost);
}

} // namespace mpmstd::core
