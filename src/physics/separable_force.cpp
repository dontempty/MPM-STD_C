#include "physics/separable_force.hpp"

#include <stdexcept>

// CPU 백엔드 — GPU 쌍둥이 separable_force.cu.

namespace mpmstd::physics {

void add_separable_force(field::Field& rhs, const field::Buffer& envelope_x,
                         const field::Buffer& shape_y, const field::Buffer& envelope_z,
                         const real_t scale, const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  if (envelope_x.size() != static_cast<std::size_t>(nx) || shape_y.size() != static_cast<std::size_t>(ny)
      || envelope_z.size() != static_cast<std::size_t>(nz))
    throw std::invalid_argument("add_separable_force: array sizes must be the local n_total of each axis");
  const real_t* ex = envelope_x.data();
  const real_t* gy = shape_y.data();
  const real_t* ez = envelope_z.data();
  const field::View r = rhs.view();
  for (int i = h; i < nx - h; ++i) {
    if (ex[i] == 0) continue;
    for (int j = h; j < ny - h; ++j) {
      const real_t factor = scale * ex[i] * gy[j];
      for (int k = h; k < nz - h; ++k) r(i, j, k) += factor * ez[k];
    }
  }
}

}  // namespace mpmstd::physics
