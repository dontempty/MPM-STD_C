#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/momentum/kernels.hpp"
#include "core/boundary_ops.hpp"   // sync_field_cpu, modify_tdma_row_cpu

#include <algorithm>
#include <cstddef>

// P1 numerics, rewired to the structural redesign (Domain + Fields + system).
// Reads velocities/nu from Fields; increments live in the MomentumSystem.

namespace mpmstd::equation {

namespace mk = momentum::kernels;

// ---- assemble: explicit BW-ADI RHS per component (T-independent) ------------
void assemble_momentum_const_visc_cpu(const core::Domain& domain, core::CpuFields& fields,
                                      core::CpuMomentumSystem& mom, real_t dt) {
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const real_t* up = fields[core::Var::U].data();
  const real_t* vp = fields[core::Var::V].data();
  const real_t* wp = fields[core::Var::W].data();
  const real_t  nu = fields.constant(core::Const::nu);
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X), *dmx1 = domain.grid.dmx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y), *dmx2 = domain.grid.dmx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z), *dmx3 = domain.grid.dmx_ptr(Direction::Z);
  const real_t conv = real_t{1};

  mk::compute_mpmstd_rhs(mom.rhs_u.data(), up, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::X, false, conv);
  mk::compute_mpmstd_rhs(mom.rhs_v.data(), vp, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::Y, false, conv);
  mk::compute_mpmstd_rhs(mom.rhs_w.data(), wp, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::Z, true,  conv);
}

void assemble_momentum_var_visc_cpu(const core::Domain&, core::CpuFields&, core::CpuMomentumSystem&, real_t) {
  /* TODO(P7): variable-viscosity cross-stress RHS for NOB/LES. */
}

// ---- one component: 3-sweep ADI (rhs_buf → out increment) -------------------
namespace {
void adi_component(core::CpuMomentumSystem& mom, std::vector<real_t>& rhs_buf,
                   const real_t* qf, const real_t* Up, const real_t* Vp, const real_t* Wp, bool z_stag,
                   const boundary::FieldBoundary& fbc, const core::Domain& domain, const core::BoundaryCondition& bc,
                   real_t nu, real_t dt, real_t* out) {
  const auto nt = mom.n_total; const auto ni = mom.n_interior;
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int i1 = ni[0], i2 = ni[1], i3 = ni[2];
  real_t* in   = rhs_buf.data();
  real_t* outb = mom.stage.data();

  for (Direction dir : bc.topology.sweep_order()) {
    const real_t* dxa  = domain.grid.dx_ptr(dir);
    const real_t* dmxa = domain.grid.dmx_ptr(dir);
    const real_t* qadv = (dir == Direction::X) ? Up : (dir == Direction::Y) ? Vp : Wp;
    const bool    zs   = (dir == Direction::Z) && z_stag;

    mk::build_adi_bands(dir, mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(),
                        in, qf, qadv, dxa, dmxa, n1, n2, n3, i1, i2, i3, nu, dt, real_t{0.5}, zs);

    int n_row = 0, n_sys = 0;
    switch (dir) {
      case Direction::X: n_row = i1; n_sys = i2 * i3; break;
      case Direction::Y: n_row = i2; n_sys = i1 * i3; break;
      case Direction::Z: n_row = i3; n_sys = i1 * i2; break;
    }
    if (bc.topology.is_periodic(dir)) {
      domain.tdma.get(dir).solve_many_cyclic(mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
    } else {
      core::modify_tdma_row_cpu(dir, fbc, domain.sub.topology().axis(dir),
                                mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
      domain.tdma.get(dir).solve_many(mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
    }
    mk::scatter_from_tdma(dir, outb, mom.D.data(), n1, n2, n3, i1, i2, i3);
    std::swap(in, outb);
  }
  std::copy(in, in + static_cast<std::size_t>(n1) * n2 * n3, out);
}
} // anonymous namespace

// ---- solve: 3-sweep ADI per component + block coupling (M2) -----------------
void solve_momentum_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::CpuFields& fields, core::CpuMomentumSystem& mom, real_t dt) {
  const real_t  nu = fields.constant(core::Const::nu);
  const real_t* Up = fields[core::Var::U].data();
  const real_t* Vp = fields[core::Var::V].data();
  const real_t* Wp = fields[core::Var::W].data();

  adi_component(mom, mom.rhs_u, Up, Up, Vp, Wp, false, bc.U, domain, bc, nu, dt, mom.dU.data());
  adi_component(mom, mom.rhs_v, Vp, Up, Vp, Wp, false, bc.V, domain, bc, nu, dt, mom.dV.data());
  adi_component(mom, mom.rhs_w, Wp, Up, Vp, Wp, true,  bc.W, domain, bc, nu, dt, mom.dW.data());

  core::sync_field_cpu(mom.dU, bc.U, domain.sub);
  core::sync_field_cpu(mom.dV, bc.V, domain.sub);
  core::sync_field_cpu(mom.dW, bc.W, domain.sub);

  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  mk::block_couple_dV(mom.dV.data(), mom.dW.data(), fields[core::Var::V].data(),
                      domain.grid.dx_ptr(Direction::Y), domain.grid.dmx_ptr(Direction::Y),
                      domain.grid.dx_ptr(Direction::Z), domain.grid.dmx_ptr(Direction::Z), n1, n2, n3, dt);
  core::sync_field_cpu(mom.dV, bc.V, domain.sub);
  mk::block_couple_dU(mom.dU.data(), mom.dV.data(), mom.dW.data(), fields[core::Var::U].data(),
                      domain.grid.dx_ptr(Direction::X), domain.grid.dmx_ptr(Direction::X),
                      domain.grid.dmx_ptr(Direction::Y), domain.grid.dmx_ptr(Direction::Z), n1, n2, n3, dt);
}

// ---- pseudo-update: q += dq (interior; caller syncs U,V,W) ------------------
void update_velocity_cpu(core::CpuFields& fields, const core::CpuMomentumSystem& mom) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  mk::add_increment(fields[core::Var::U].data(), mom.dU.data(), n1, n2, n3);
  mk::add_increment(fields[core::Var::V].data(), mom.dV.data(), n1, n2, n3);
  mk::add_increment(fields[core::Var::W].data(), mom.dW.data(), n1, n2, n3);
}

} // namespace mpmstd::equation
