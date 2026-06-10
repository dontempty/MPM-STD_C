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

// NOB buoyancy: F = invRho·(θ + a12pera11·θ²·DeltaT) on the gravity-axis
// component. Cell-centred (matches the OB code's cell-T-on-W convention), so
// with invRho≡1, a12pera11=0 it reduces to add_buoyancy_force_cpu(coeff=1,T_ref=0).
void add_buoyancy_force_nob_cpu(core::CpuMomentumSystem& mom, const core::CpuField& T,
                                const core::CpuField& inv_rho, const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* t  = T.data();
  const real_t* ir = inv_rho.data();
  std::vector<real_t>& rhs_vec = (p.gravity_axis == 0) ? mom.rhs_u
                               : (p.gravity_axis == 1) ? mom.rhs_v : mom.rhs_w;
  real_t* rhs = rhs_vec.data();
  const real_t a12pera11 = p.a12pera11, dT = p.delta_T;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
        const real_t th = t[c];
        rhs[c] += dt * ir[c] * (th + a12pera11 * th * th * dT);
      }
}

} // namespace mpmstd::physics
