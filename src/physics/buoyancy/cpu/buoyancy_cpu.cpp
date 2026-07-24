#include "physics/buoyancy/buoyancy.hpp"
#include "common/macros.hpp"
#include "common/direction.hpp"

// OB Boussinesq buoyancy modifier. Composed in main AFTER momentum assemble,
// BEFORE solve: adds the body force F = coeff*(T - T_ref) to the gravity-axis
// component's explicit RHS (rhs += dt*F), matching the validated src/ scheme
// (momentum source_name = "T"). gravity_axis picks the component: DHVC = x
// (streamwise), RBC = z (wall-normal). coeff = 1 for the free-fall nondim.

namespace mpmstd::physics::buoyancy {

void add_buoyancy_force_constant_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom,
                            const core::CpuField& T, const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* t = T.data();
  std::vector<real_t>& rhs_vec = (p.gravity_axis == 0) ? mom.rhs_u
                               : (p.gravity_axis == 1) ? mom.rhs_v : mom.rhs_w;
  real_t* rhs = rhs_vec.data();
  const real_t coeff = p.coeff, t_ref = p.T_ref;
  // PaScaL_TCS f90:348 — interpolate T onto the gravity-axis component face:
  //   Tc = 0.5*(dxg[ig-1]*T[c] + dxg[ig]*T[c-gstride])/dmxg[ig]
  const Direction gd = static_cast<Direction>(p.gravity_axis);
  const real_t* dxg  = domain.grid.dx_ptr(gd);
  const real_t* dmxg = domain.grid.dmx_ptr(gd);
  const int gstride = (p.gravity_axis == 0) ? n2 * n3 : (p.gravity_axis == 1) ? n3 : 1;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
        const int ig = (p.gravity_axis == 0) ? i : (p.gravity_axis == 1) ? j : k;
        const real_t Tc = real_t{0.5}*(dxg[ig-1]*t[c] + dxg[ig]*t[c - gstride])/dmxg[ig];
        rhs[c] += dt * coeff * (Tc - t_ref);
      }
}

// NOB buoyancy: F = invRhoc·(Tc + a12pera11·Tc²·DeltaT) on the gravity-axis
// component, with both T and invRho interpolated onto that component's
// staggered face (PaScaL_TCS mom.f90:711/731, module_solve_momentum solvedV) —
// same face-interpolation as add_buoyancy_force_constant_property_cpu above, extended to invRho.
// With invRho≡1, a12pera11=0 it reduces to add_buoyancy_force_constant_property_cpu(coeff=1,T_ref=0).
void add_buoyancy_force_variable_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom,
                                const core::CpuField& T, const core::CpuField& inv_rho,
                                const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* t  = T.data();
  const real_t* ir = inv_rho.data();
  std::vector<real_t>& rhs_vec = (p.gravity_axis == 0) ? mom.rhs_u
                               : (p.gravity_axis == 1) ? mom.rhs_v : mom.rhs_w;
  real_t* rhs = rhs_vec.data();
  const real_t a12pera11 = p.a12pera11, dT = p.delta_T;
  const Direction gd = static_cast<Direction>(p.gravity_axis);
  const real_t* dxg  = domain.grid.dx_ptr(gd);
  const real_t* dmxg = domain.grid.dmx_ptr(gd);
  const int gstride = (p.gravity_axis == 0) ? n2 * n3 : (p.gravity_axis == 1) ? n3 : 1;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
        const int ig = (p.gravity_axis == 0) ? i : (p.gravity_axis == 1) ? j : k;
        const real_t Tc  = real_t{0.5}*(dxg[ig-1]*t[c]  + dxg[ig]*t[c - gstride])/dmxg[ig];
        const real_t irc = real_t{0.5}*(dxg[ig-1]*ir[c] + dxg[ig]*ir[c - gstride])/dmxg[ig];
        rhs[c] += dt * irc * (Tc + a12pera11 * Tc * Tc * dT);
      }
}

} // namespace mpmstd::physics::buoyancy
