#pragma once

#include "common/types.hpp"
#include <vector>

namespace mpmstd::grid {

// Stretching modes per axis.
enum class StretchKind {
  Uniform,
  Tanh,    // hyperbolic-tangent clustering, both walls
};

// Generate global cell-face coordinates (size = n_cells + 1) along [0, L]
// according to `kind`. For `Tanh`, `gamma` controls clustering intensity
// (gamma >> 1 packs cells near both walls).
//
// Convention: faces[0] = 0, faces[n_cells] = L.
std::vector<real_t> make_face_coordinates(StretchKind kind,
                                           int n_cells,
                                           real_t L,
                                           real_t gamma = 0.0);

} // namespace mpmstd::grid
