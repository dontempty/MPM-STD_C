#include "grid/stretching.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mpmstd::grid {

namespace {

void check_common(const char* who, int n_cells, real_t length) {
  if (n_cells <= 0)
    throw std::invalid_argument(std::string(who) + ": n_cells must be > 0, got "
                                + std::to_string(n_cells));
  if (!(length > 0))
    throw std::invalid_argument(std::string(who) + ": length must be > 0, got "
                                + std::to_string(length));
}

}  // namespace

std::vector<real_t> make_uniform_faces(int n_cells, real_t length) {
  check_common("make_uniform_faces", n_cells, length);
  std::vector<real_t> faces(n_cells + 1);
  for (int i = 0; i <= n_cells; ++i)
    faces[i] = length * static_cast<real_t>(i) / static_cast<real_t>(n_cells);
  faces.front() = 0;
  faces.back() = length;
  return faces;
}

std::vector<real_t> make_tanh_faces(int n_cells, real_t length, real_t gamma) {
  check_common("make_tanh_faces", n_cells, length);
  if (!(gamma > 0))
    throw std::invalid_argument("make_tanh_faces: gamma must be > 0, got "
                                + std::to_string(gamma));

  const real_t half_gamma = real_t{0.5} * gamma;
  const real_t denom = std::tanh(half_gamma);
  std::vector<real_t> faces(n_cells + 1);
  for (int i = 0; i <= n_cells; ++i) {
    const real_t xi = static_cast<real_t>(i) / static_cast<real_t>(n_cells);
    faces[i] = real_t{0.5} * length
               * (real_t{1} + std::tanh(half_gamma * (real_t{2} * xi - real_t{1})) / denom);
  }
  faces.front() = 0;       // 끝점 강제 — 부동소수 드리프트 방지
  faces.back() = length;
  return faces;
}

std::vector<real_t> make_tanh_wall_faces(int n_cells, real_t length, real_t gamma) {
  check_common("make_tanh_wall_faces", n_cells, length);
  if (!(gamma > 0))
    throw std::invalid_argument("make_tanh_wall_faces: gamma must be > 0, got "
                                + std::to_string(gamma));
  const real_t denom = std::tanh(gamma);
  std::vector<real_t> faces(n_cells + 1);
  for (int i = 0; i <= n_cells; ++i) {
    const real_t xi = static_cast<real_t>(i) / static_cast<real_t>(n_cells);
    faces[i] = length * (real_t{1} - std::tanh(gamma * (real_t{1} - xi)) / denom);
  }
  faces.front() = 0;
  faces.back() = length;
  return faces;
}

}  // namespace mpmstd::grid
