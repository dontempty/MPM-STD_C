#include "physics/separable_force.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"

// GPU 백엔드 — separable_force.cpp 의 쌍둥이.

namespace mpmstd::physics {

namespace {

__global__ void separable_force_kernel(const field::View r, const real_t* ex, const real_t* gy,
                                       const real_t* ez, const real_t scale, const int h,
                                       const int i1, const int i2, const int i3) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  r(i, j, k) += scale * ex[i] * gy[j] * ez[k];
}

}  // namespace

void add_separable_force(field::Field& rhs, const field::Buffer& envelope_x,
                         const field::Buffer& shape_y, const field::Buffer& envelope_z,
                         const real_t scale, const parallel::Subdomain& sub) {
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  if (envelope_x.size() != static_cast<std::size_t>(nx) || shape_y.size() != static_cast<std::size_t>(ny)
      || envelope_z.size() != static_cast<std::size_t>(nz))
    throw std::invalid_argument("add_separable_force: array sizes must be the local n_total of each axis");
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  const int threads = 256;
  separable_force_kernel<<<static_cast<int>((n + threads - 1) / threads), threads>>>(
      rhs.view(), envelope_x.data(), shape_y.data(), envelope_z.data(), scale, halo_width, i1, i2,
      i3);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::physics
