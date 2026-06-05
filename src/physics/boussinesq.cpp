#include "physics/boussinesq.hpp"
#include "common/macros.hpp"

#include <cmath>

namespace mpmstd::physics {

real_t BoussinesqParams::viscosity()   const { return std::sqrt(Pr / Ra); }
real_t BoussinesqParams::diffusivity() const { return 1.0 / std::sqrt(Ra * Pr); }

void compute_z_buoyancy(field::ScalarField&       src,
                         const field::ScalarField& T) {
  const std::size_t n = src.n_elements();
  real_t*       s_ptr = src.host_ptr();
  const real_t* t_ptr = T.host_ptr();
  for (std::size_t i = 0; i < n; ++i)
    s_ptr[i] = t_ptr[i];
}

} // namespace mpmstd::physics
