#include "grid/stretching.hpp"

#include <cmath>
#include <stdexcept>

namespace mpmstd::grid {

std::vector<real_t> make_face_coordinates(StretchKind kind,
                                           int n_cells,
                                           real_t L,
                                           real_t gamma) {
  if (n_cells <= 0) {
    throw std::invalid_argument("make_face_coordinates: n_cells must be > 0");
  }
  if (L <= 0.0) {
    throw std::invalid_argument("make_face_coordinates: L must be > 0");
  }

  std::vector<real_t> faces(static_cast<std::size_t>(n_cells) + 1);

  switch (kind) {
    case StretchKind::Uniform: {
      const real_t dx = L / static_cast<real_t>(n_cells);
      for (int i = 0; i <= n_cells; ++i) {
        faces[i] = dx * static_cast<real_t>(i);
      }
      break;
    }
    case StretchKind::Tanh: {
      // PaScaL_TCS style: x(xi) = L/2 * (1 + tanh(0.5*gamma*(2*xi/Nm - 1)) / tanh(0.5*gamma))
      // where xi = 0..Nm. Result clusters near both walls (xi=0 and xi=Nm).
      if (gamma <= 0.0) {
        throw std::invalid_argument("make_face_coordinates: tanh requires gamma > 0");
      }
      const real_t half_gamma  = 0.5 * gamma;
      const real_t denom_tanh  = std::tanh(half_gamma);
      for (int i = 0; i <= n_cells; ++i) {
        const real_t xi   = static_cast<real_t>(i) / static_cast<real_t>(n_cells);
        const real_t arg  = half_gamma * (2.0 * xi - 1.0);
        faces[i] = 0.5 * L * (1.0 + std::tanh(arg) / denom_tanh);
      }
      // Enforce exact endpoints to avoid floating-point drift.
      faces.front() = 0.0;
      faces.back()  = L;
      break;
    }
  }

  return faces;
}

} // namespace mpmstd::grid
