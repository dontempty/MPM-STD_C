#include "field/gpu/field.hpp"

namespace mpmstd::field {

namespace {

__global__ void fill_kernel(real_t* p, const long long n, const real_t value) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q < n) p[q] = value;
}

}  // namespace

void Field::fill(const real_t value) {
  const long long n = static_cast<long long>(size_);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  fill_kernel<<<blocks, threads>>>(data_, n, value);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::field
