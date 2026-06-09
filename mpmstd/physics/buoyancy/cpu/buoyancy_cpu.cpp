#include "physics/buoyancy/buoyancy.hpp"
#include "common/macros.hpp"

// OB Boussinesq buoyancy modifier. Composed in main AFTER momentum assemble,
// BEFORE solve: adds the body force F = coeff*(T - T_ref) to the gravity-axis
// component's explicit RHS (rhs += dt*F), matching the validated src/ scheme
// (momentum source_name = "T"). gravity_axis picks the component: DHVC = x
// (streamwise), RBC = z (wall-normal). coeff = 1 for the free-fall nondim.

namespace mpmstd::physics {

void add_buoyancy_force_cpu(core::CpuMomentumSystem& mom, const core::CpuField& T,
                            const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* t = T.data();
  std::vector<real_t>& rhs_vec = (p.gravity_axis == 0) ? mom.rhs_u
                               : (p.gravity_axis == 1) ? mom.rhs_v : mom.rhs_w;
  real_t* rhs = rhs_vec.data();
  const real_t coeff = p.coeff, t_ref = p.T_ref;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
        rhs[c] += dt * coeff * (t[c] - t_ref);
      }
}

} // namespace mpmstd::physics
