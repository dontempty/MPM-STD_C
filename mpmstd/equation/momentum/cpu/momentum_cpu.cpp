#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/momentum/kernels.hpp"
#include "core/halo.hpp"            // exchange_halo_cpu
#include "core/boundary_ops.hpp"    // apply_ghost_cpu, modify_tdma_row_cpu

#include <algorithm>                // std::copy, std::swap
#include <cstddef>

// P1 — CPU momentum free functions, calling the validated MPM-STD kernels
// (copied into mpmstd/equation/momentum/). Faithful translation of the old
// MomentumEquation (predict + block coupling + apply) into free functions.

namespace mpmstd::equation {

namespace mk = momentum::kernels;

// ---- assemble: explicit BW-ADI RHS per component (T-independent) ------------
void assemble_momentum_const_visc_cpu(core::MomentumSystem& mom,
                                      const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                                      const core::Grid& grid, real_t nu, real_t dt) {
  mom.allocate(U.n_total(), U.n_interior());
  const auto nt = U.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const real_t* up = U.data();  const real_t* vp = V.data();  const real_t* wp = W.data();
  const real_t* dx1 = grid.dx_ptr(Direction::X), *dmx1 = grid.dmx_ptr(Direction::X);
  const real_t* dx2 = grid.dx_ptr(Direction::Y), *dmx2 = grid.dmx_ptr(Direction::Y);
  const real_t* dx3 = grid.dx_ptr(Direction::Z), *dmx3 = grid.dmx_ptr(Direction::Z);
  const real_t conv = real_t{1};   // conv_f=1.0 (full physical convection; see kernels)

  mk::compute_mpmstd_rhs(mom.rhs_u.data(), up, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::X, false, conv);
  mk::compute_mpmstd_rhs(mom.rhs_v.data(), vp, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::Y, false, conv);
  mk::compute_mpmstd_rhs(mom.rhs_w.data(), wp, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::Z, true,  conv);
}

void assemble_momentum_var_visc_cpu(core::MomentumSystem&, const core::CpuField&, const core::CpuField&,
                                    const core::CpuField&, const core::CpuField&, const core::Grid&, real_t) {
  /* TODO(P7): full variable-viscosity cross-stress RHS for NOB/LES. */
}

// ---- one component: 3-sweep ADI  (rhs_buf → out increment) ------------------
namespace {
void adi_component(core::MomentumSystem& mom, std::vector<real_t>& rhs_buf,
                   const real_t* qf, const real_t* Up, const real_t* Vp, const real_t* Wp,
                   bool z_stag, const boundary::FieldBoundary& fbc,
                   const core::Grid& grid, const core::Boundary& problem,
                   linear_solver::tdma::TdmaRegistry& tdma, const core::Subdomain& sub,
                   real_t nu, real_t dt, real_t* out) {
  const auto nt = mom.n_total; const auto ni = mom.n_interior;
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int i1 = ni[0], i2 = ni[1], i3 = ni[2];

  real_t* in   = rhs_buf.data();
  real_t* outb = mom.stage.data();

  for (Direction d : problem.topology.sweep_order()) {
    const real_t* dxa  = grid.dx_ptr(d);
    const real_t* dmxa = grid.dmx_ptr(d);
    const real_t* qadv = (d == Direction::X) ? Up : (d == Direction::Y) ? Vp : Wp;
    const bool    zs   = (d == Direction::Z) && z_stag;

    mk::build_adi_bands(d, mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(),
                        in, qf, qadv, dxa, dmxa, n1, n2, n3, i1, i2, i3,
                        nu, dt, real_t{0.5}, zs);

    int n_row = 0, n_sys = 0;
    switch (d) {
      case Direction::X: n_row = i1; n_sys = i2 * i3; break;
      case Direction::Y: n_row = i2; n_sys = i1 * i3; break;
      case Direction::Z: n_row = i3; n_sys = i1 * i2; break;
    }

    if (problem.topology.is_periodic(d)) {
      tdma.get(d).solve_many_cyclic(mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
    } else {
      core::modify_tdma_row_cpu(d, fbc, sub.topology().axis(d),
                                mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
      tdma.get(d).solve_many(mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
    }

    mk::scatter_from_tdma(d, outb, mom.D.data(), n1, n2, n3, i1, i2, i3);
    std::swap(in, outb);
  }
  std::copy(in, in + static_cast<std::size_t>(n1) * n2 * n3, out);
}
} // anonymous namespace

// ---- solve: 3-sweep ADI per component + block coupling (M2) -----------------
void solve_momentum_cpu(core::MomentumSystem& mom,
                        const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                        core::CpuField& dU, core::CpuField& dV, core::CpuField& dW,
                        const core::Grid& grid, const core::Boundary& problem,
                        linear_solver::tdma::TdmaRegistry& tdma, const core::Subdomain& sub,
                        real_t nu, real_t dt) {
  const real_t* Up = U.data(); const real_t* Vp = V.data(); const real_t* Wp = W.data();

  // (1) per-component diagonal ADI predictors → increments dU,dV,dW
  adi_component(mom, mom.rhs_u, Up, Up, Vp, Wp, false, problem.U, grid, problem, tdma, sub, nu, dt, dU.data());
  adi_component(mom, mom.rhs_v, Vp, Up, Vp, Wp, false, problem.V, grid, problem, tdma, sub, nu, dt, dV.data());
  adi_component(mom, mom.rhs_w, Wp, Up, Vp, Wp, true,  problem.W, grid, problem, tdma, sub, nu, dt, dW.data());

  // increments need halos+ghosts before the block-coupling stencil
  core::exchange_halo_cpu(dU, sub); core::apply_ghost_cpu(dU, problem.U, sub);
  core::exchange_halo_cpu(dV, sub); core::apply_ghost_cpu(dV, problem.V, sub);
  core::exchange_halo_cpu(dW, sub); core::apply_ghost_cpu(dW, problem.W, sub);

  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];

  // (2) blockLdV: dV -= dt*0.25*(dW·∂V/∂z)
  mk::block_couple_dV(dV.data(), dW.data(), V.data(),
                      grid.dx_ptr(Direction::Y), grid.dmx_ptr(Direction::Y),
                      grid.dx_ptr(Direction::Z), grid.dmx_ptr(Direction::Z),
                      n1, n2, n3, dt);
  core::exchange_halo_cpu(dV, sub); core::apply_ghost_cpu(dV, problem.V, sub);

  // (3) blockLdU: dU -= dt*0.25*(dV·∂U/∂y + dW·∂U/∂z)  (uses corrected dV)
  mk::block_couple_dU(dU.data(), dV.data(), dW.data(), U.data(),
                      grid.dx_ptr(Direction::X), grid.dmx_ptr(Direction::X),
                      grid.dmx_ptr(Direction::Y), grid.dmx_ptr(Direction::Z),
                      n1, n2, n3, dt);
}

// ---- pseudo-update: q += dq (interior; caller does halo + BC ghost) ---------
void update_velocity_cpu(core::CpuField& U, core::CpuField& V, core::CpuField& W,
                         const core::CpuField& dU, const core::CpuField& dV, const core::CpuField& dW) {
  const auto nt = U.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  mk::add_increment(U.data(), dU.data(), n1, n2, n3);
  mk::add_increment(V.data(), dV.data(), n1, n2, n3);
  mk::add_increment(W.data(), dW.data(), n1, n2, n3);
}

} // namespace mpmstd::equation
