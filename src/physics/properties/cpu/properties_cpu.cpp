#include "physics/properties/properties.hpp"
#include "common/macros.hpp"

#include <cstddef>

// NOB glycerol property update, ported from the reference PaScaL_TCS
// (module_global.f90 coefficients; module_solve_momentum/thermal evaluation).
// Properties are polynomials in Δθ = DeltaT·θ. Computed over the FULL array
// (incl. halo, since T is already synced) so the face interpolations + ghosts
// downstream are valid.

namespace mpmstd::physics::properties {

// The property formulas themselves live in properties.hpp
// (evaluate_properties, shared with the gpu/ backend); this file only loops it
// over the field.
void update_properties_cpu(CpuPropertyFields& p, const core::CpuField& T, const PropertyModel& m) {
  const auto nt = T.n_total();
  const std::size_t n = static_cast<std::size_t>(nt[0]) * nt[1] * nt[2];
  const real_t* Tp = T.data();
  real_t* Mu = p.Mu.data();             real_t* invRho = p.invRho.data();
  real_t* KPP = p.KPP.data();           real_t* dKPP = p.dKPP.data();
  real_t* invRhoCp = p.invRhoCp.data(); real_t* dinvRhoCp = p.dinvRhoCp.data();

  for (std::size_t i = 0; i < n; ++i) {
    const PropertyValues v = evaluate_properties(m, Tp[i]);
    Mu[i] = v.Mu;             invRho[i] = v.invRho;
    KPP[i] = v.KPP;           dKPP[i] = v.dKPP;
    invRhoCp[i] = v.invRhoCp; dinvRhoCp[i] = v.dinvRhoCp;
  }
}

} // namespace mpmstd::physics::properties
