#include "field/scalar_field.hpp"
#include "parallel/cuda/cuda_memory.hpp"

#include <algorithm>
#include <cstring>

namespace mpmstd::field {

ScalarField::ScalarField(const parallel::mpi::Subdomain& sub,
                          parallel::Backend& backend,
                          std::string name)
  : sub_(sub), backend_(backend), name_(std::move(name)),
    n_elements_(sub.n_elements()),
    host_(sub.n_elements(), 0.0) {

#ifdef MPMSTD_BACKEND_CUDA
  device_ = reinterpret_cast<real_t*>(backend_.alloc(n_elements_ * sizeof(real_t)));
  // Initialise device buffer to zero (matches host's value-init).
  // We avoid cudaMemset because we want full bitwise zero of `real_t`, which
  // for IEEE float/double *is* all-zero-bytes, so memset works:
  parallel::cuda_helpers::copy_host_to_device(device_, host_.data(),
                                                n_elements_ * sizeof(real_t));
#else
  // CPU build: device pointer aliases host buffer. No separate alloc.
  device_ = host_.data();
#endif
}

ScalarField::~ScalarField() {
#ifdef MPMSTD_BACKEND_CUDA
  if (device_) {
    backend_.free(device_);
    device_ = nullptr;
  }
#endif
}

void ScalarField::fill_host(real_t value) {
  std::fill(host_.begin(), host_.end(), value);
}

void ScalarField::copy_from_host(const real_t* src) {
  std::memcpy(host_.data(), src, n_elements_ * sizeof(real_t));
}

real_t* ScalarField::device_ptr() {
#ifdef MPMSTD_BACKEND_CUDA
  return device_;
#else
  return host_.data();
#endif
}
const real_t* ScalarField::device_ptr() const {
#ifdef MPMSTD_BACKEND_CUDA
  return device_;
#else
  return host_.data();
#endif
}

void ScalarField::to_device() {
#ifdef MPMSTD_BACKEND_CUDA
  parallel::cuda_helpers::copy_host_to_device(device_, host_.data(),
                                                n_elements_ * sizeof(real_t));
#endif
}

void ScalarField::to_host() {
#ifdef MPMSTD_BACKEND_CUDA
  parallel::cuda_helpers::copy_device_to_host(host_.data(), device_,
                                                n_elements_ * sizeof(real_t));
#endif
}

void ScalarField::exchange_halo() {
  // On CPU build: exchange host buffer.
  // On CUDA build with CUDA-aware MPI: exchange device buffer directly.
  // (Without CUDA-aware MPI: caller must to_host() / exchange / to_device(),
  //  but that path is not yet implemented — see cuda_aware_mpi.hpp.)
  real_t* buf =
#ifdef MPMSTD_BACKEND_CUDA
    device_;
#else
    host_.data();
#endif
  sub_.exchange_halo(buf);
}

} // namespace mpmstd::field
