// Placeholder — GPU implementation deferred to M5'.

#include "equation/momentum/kernels/kernels.hpp"
#include <stdexcept>

namespace mpmstd::equation::momentum::kernels {

void laplacian_explicit_rhs(real_t*, const real_t*,
                              const real_t*, const real_t*,
                              const real_t*, const real_t*,
                              const real_t*, const real_t*,
                              int, int, int, real_t) {
  throw std::runtime_error("momentum::kernels::laplacian_explicit_rhs: CUDA deferred to M5'");
}

void add_convection_rhs(real_t*, const real_t*,
                         const real_t*, const real_t*, const real_t*,
                         const real_t*, const real_t*, const real_t*,
                         int, int, int, real_t) {
  throw std::runtime_error("momentum::kernels::add_convection_rhs: CUDA deferred to M5'");
}

void add_source_rhs(real_t*, const real_t*, int, int, int, real_t) {
  throw std::runtime_error("momentum::kernels::add_source_rhs: CUDA deferred to M5'");
}

void build_adi_bands(Direction,
                      real_t*, real_t*, real_t*, real_t*,
                      const real_t*,
                      const real_t*, const real_t*,
                      int, int, int, int, int, int,
                      real_t, real_t) {
  throw std::runtime_error("momentum::kernels::build_adi_bands: CUDA deferred to M5'");
}

void scatter_from_tdma(Direction,
                        real_t*, const real_t*,
                        int, int, int, int, int, int) {
  throw std::runtime_error("momentum::kernels::scatter_from_tdma: CUDA deferred to M5'");
}

void add_increment(real_t*, const real_t*, int, int, int) {
  throw std::runtime_error("momentum::kernels::add_increment: CUDA deferred to M5'");
}

} // namespace mpmstd::equation::momentum::kernels
