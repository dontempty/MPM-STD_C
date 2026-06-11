#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/scalar/kernels.hpp"
#include "core/boundary_ops.hpp"   // modify_tdma_row_cpu

#include <algorithm>
#include <cstddef>

// Scalar (energy) transport ported from the validated src/equation/scalar
// (byte-identical kernels) onto the structural-redesign API (Domain + Fields +
// ScalarSystem). OB constant diffusivity; CN delta-form, 3-sweep ADI.

namespace mpmstd::equation {

namespace sk = scalar::kernels;

// ---- assemble: explicit CN RHS (Laplacian + convection) ---------------------
void assemble_scalar_const_diff_cpu(const core::Domain& domain, core::CpuFields& fields,
                                    core::CpuScalarSystem& scalar_system, real_t dt) {
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const real_t* T     = fields[core::Var::T].data();
  const real_t  alpha = fields.constant(core::Const::alpha_T);

  // (1) explicit full Laplacian: rhs = dt*alpha*(Lx+Ly+Lz) T
  sk::laplacian_explicit_rhs(scalar_system.rhs.data(), T,
      domain.grid.dx_ptr(Direction::X), domain.grid.dmx_ptr(Direction::X),
      domain.grid.dx_ptr(Direction::Y), domain.grid.dmx_ptr(Direction::Y),
      domain.grid.dx_ptr(Direction::Z), domain.grid.dmx_ptr(Direction::Z),
      n1, n2, n3, dt * alpha);

  // (1b) convection: rhs += -dt * u·∇T (conservative flux, cell-centred T, face vel)
  sk::add_convection_rhs(scalar_system.rhs.data(), T,
      fields[core::Var::U].data(), fields[core::Var::V].data(), fields[core::Var::W].data(),
      domain.grid.dx_ptr(Direction::X), domain.grid.dx_ptr(Direction::Y), domain.grid.dx_ptr(Direction::Z),
      n1, n2, n3, -dt);
}

// ---- solve: 3-sweep ADI (sweep order from BC) + T += increment --------------
void solve_scalar_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                      core::CpuFields& fields, core::CpuScalarSystem& sys, real_t dt) {
  const auto nt = sys.n_total; const auto ni = sys.n_interior;
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int i1 = ni[0], i2 = ni[1], i3 = ni[2];
  const real_t alpha = fields.constant(core::Const::alpha_T);
  const boundary::FieldBoundary& fbc = bc.T;

  real_t* in   = sys.rhs.data();
  real_t* outb = sys.stage.data();
  for (Direction d : bc.topology.sweep_order()) {
    sk::build_adi_bands(d, sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(),
                        in, domain.grid.dx_ptr(d), domain.grid.dmx_ptr(d),
                        n1, n2, n3, i1, i2, i3, alpha, dt);

    int n_row = 0, n_sys = 0;
    switch (d) {
      case Direction::X: n_row = i1; n_sys = i2 * i3; break;
      case Direction::Y: n_row = i2; n_sys = i1 * i3; break;
      case Direction::Z: n_row = i3; n_sys = i1 * i2; break;
    }
    if (bc.topology.is_periodic(d)) {
      domain.tdma.get(d).solve_many_cyclic(sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(), n_sys, n_row);
    } else {
      core::modify_tdma_row_cpu(d, fbc, domain.sub.topology().axis(d),
                                sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(), n_sys, n_row);
      domain.tdma.get(d).solve_many(sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(), n_sys, n_row);
    }
    sk::scatter_from_tdma(d, outb, sys.D.data(), n1, n2, n3, i1, i2, i3);
    std::swap(in, outb);
  }
  // `in` holds the final increment δ; T^{n+1} = T^n + δ (caller syncs T).
  sk::add_increment(fields[core::Var::T].data(), in, n1, n2, n3);
}

// =============================================================================
//  Variable-κ̂ (NOB) ENERGY — FAITHFUL port of PaScaL_TCS module_solve_thermal
//  (delta-form 3-factor ADI). The assemble builds the explicit residual R (now
//  INCLUDING convection, 0.5-skew advective); the solve builds the bands per
//  sweep (stage-1 packs M·θⁿ + dt-scales D; later stages copy D through) with
//  the semi-implicit conservative-face convection inside each band. dt is
//  applied inside the band pack (ref line 358), so assemble does NOT scale by dt.
//  Wall: ghost + modify_tdma_row_cpu fold (validated mpmstd wiring), δθ_wall=0.
//  With κ̂≡1, dκ̂≡0, invρĈp≡1, dinvρĈp≡0 the diffusion reduces to the const
//  Laplacian; the semi-implicit convection is the faithful upgrade.
// =============================================================================
void assemble_scalar_var_diff_cpu(const core::Domain& domain, core::CpuFields& fields,
                                  const core::CpuField& kappa, const core::CpuField& dkappa,
                                  const core::CpuField& inv_rhocp, const core::CpuField& dinv_rhocp,
                                  core::CpuScalarSystem& scalar_system, real_t dt) {
  (void)dt;   // dt is applied inside the stage-1 band pack (ref 358), not here
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const real_t  Ct = fields.constant(core::Const::alpha_T);   // = 1/√(Ra·Pr)

  // Full explicit residual R = −0.5·u·∇θ + 0.5·Ct·invρĈp·∇·(κ̂∇θ)|dκ̂-lin
  //                            − 0.5·Ct·dinvρĈp·θ·∇·(κ̂∇θ)
  sk::energy_residual_rhs_var(scalar_system.rhs.data(), fields[core::Var::T].data(),
      fields[core::Var::U].data(), fields[core::Var::V].data(), fields[core::Var::W].data(),
      kappa.data(), dkappa.data(), inv_rhocp.data(), dinv_rhocp.data(),
      domain.grid.dx_ptr(Direction::X), domain.grid.dmx_ptr(Direction::X),
      domain.grid.dx_ptr(Direction::Y), domain.grid.dmx_ptr(Direction::Y),
      domain.grid.dx_ptr(Direction::Z), domain.grid.dmx_ptr(Direction::Z),
      n1, n2, n3, Ct);
}

void solve_scalar_var_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                          core::CpuFields& fields,
                          const core::CpuField& kappa, const core::CpuField& dkappa,
                          const core::CpuField& inv_rhocp, const core::CpuField& dinv_rhocp,
                          core::CpuScalarSystem& sys, real_t dt) {
  const auto nt = sys.n_total; const auto ni = sys.n_interior;
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int i1 = ni[0], i2 = ni[1], i3 = ni[2];
  const real_t Ct = fields.constant(core::Const::alpha_T);
  const boundary::FieldBoundary& fbc = bc.T;

  const real_t* T  = fields[core::Var::T].data();
  const real_t* U  = fields[core::Var::U].data();
  const real_t* V  = fields[core::Var::V].data();
  const real_t* W  = fields[core::Var::W].data();
  const real_t* kp = kappa.data();    const real_t* dkp = dkappa.data();
  const real_t* ir = inv_rhocp.data(); const real_t* dir = dinv_rhocp.data();
  const real_t* dx1=domain.grid.dx_ptr(Direction::X), *dmx1=domain.grid.dmx_ptr(Direction::X);
  const real_t* dx2=domain.grid.dx_ptr(Direction::Y), *dmx2=domain.grid.dmx_ptr(Direction::Y);
  const real_t* dx3=domain.grid.dx_ptr(Direction::Z), *dmx3=domain.grid.dmx_ptr(Direction::Z);

  real_t* in   = sys.rhs.data();      // holds R on entry; then the running increment
  real_t* outb = sys.stage.data();
  bool first = true;
  for (Direction d : bc.topology.sweep_order()) {
    sk::build_energy_adi_bands_var(
        d, sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(),
        in, T, U, V, W, kp, dkp, ir, dir,
        dx1, dmx1, dx2, dmx2, dx3, dmx3,
        n1, n2, n3, i1, i2, i3, Ct, dt, /*is_stage1=*/first);

    int n_row = 0, n_sys = 0;
    switch (d) {
      case Direction::X: n_row = i1; n_sys = i2 * i3; break;
      case Direction::Y: n_row = i2; n_sys = i1 * i3; break;
      case Direction::Z: n_row = i3; n_sys = i1 * i2; break;
    }
    if (bc.topology.is_periodic(d)) {
      domain.tdma.get(d).solve_many_cyclic(sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(), n_sys, n_row);
    } else {
      core::modify_tdma_row_cpu(d, fbc, domain.sub.topology().axis(d),
                                sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(), n_sys, n_row);
      domain.tdma.get(d).solve_many(sys.A.data(), sys.B.data(), sys.C.data(), sys.D.data(), n_sys, n_row);
    }
    sk::scatter_from_tdma(d, outb, sys.D.data(), n1, n2, n3, i1, i2, i3);
    std::swap(in, outb);
    first = false;
  }
  // `in` holds δθ; θ^{n+1} = θ^n + δθ (caller syncs T).
  sk::add_increment(fields[core::Var::T].data(), in, n1, n2, n3);
}

} // namespace mpmstd::equation
