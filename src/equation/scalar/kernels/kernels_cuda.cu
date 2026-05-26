// Placeholder. The real GPU implementation lands in M5'.
//
// For now we keep the file present so that the build matrix is symmetric —
// when the project is built with BACKEND=cuda this is the file that gets
// compiled instead of kernels_cpu.cpp.

#include "equation/scalar/kernels/kernels.hpp"
#include "common/macros.hpp"
#include <stdexcept>

namespace mpmstd::equation::scalar::kernels {

void laplacian_explicit_rhs(real_t*, const real_t*,
                              const real_t*, const real_t*,
                              const real_t*, const real_t*,
                              const real_t*, const real_t*,
                              int, int, int, real_t) {
  throw std::runtime_error("scalar::kernels::laplacian_explicit_rhs: CUDA implementation deferred to M5'");
}

void add_convection_rhs(real_t*, const real_t*,
                          const real_t*, const real_t*, const real_t*,
                          const real_t*, const real_t*, const real_t*,
                          int, int, int, real_t) {
  throw std::runtime_error("scalar::kernels::add_convection_rhs: CUDA implementation deferred to M5'");
}

void build_adi_bands(Direction,
                     real_t*, real_t*, real_t*, real_t*,
                     const real_t*,
                     const real_t*, const real_t*,
                     int, int, int, int, int, int,
                     real_t, real_t) {
  throw std::runtime_error("scalar::kernels::build_adi_bands: CUDA implementation deferred to M5'");
}

void scatter_from_tdma(Direction,
                        real_t*,
                        const real_t*,
                        int, int, int, int, int, int) {
  throw std::runtime_error("scalar::kernels::scatter_from_tdma: CUDA implementation deferred to M5'");
}

void add_increment(real_t*, const real_t*, int, int, int) {
  throw std::runtime_error("scalar::kernels::add_increment: CUDA implementation deferred to M5'");
}

} // namespace mpmstd::equation::scalar::kernels
