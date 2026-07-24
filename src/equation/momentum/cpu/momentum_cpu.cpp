#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/momentum/kernels.hpp"
#include "core/boundary_ops.hpp"   // sync_field_cpu, modify_tdma_row_cpu

#include <mpi.h>

#include <algorithm>
#include <cstddef>

// P1 numerics, rewired to the structural redesign (Domain + Fields + system).
// Reads velocities/nu from Fields; increments live in the MomentumSystem.

namespace mpmstd::equation::momentum {


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

  kernels::compute_mpmstd_rhs(mom.rhs_u.data(), up, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::X, false, conv);
  kernels::compute_mpmstd_rhs(mom.rhs_v.data(), vp, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::Y, false, conv);
  kernels::compute_mpmstd_rhs(mom.rhs_w.data(), wp, up, vp, wp,
                         dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, nu, dt, Direction::Z, true,  conv);
}

// Variable-viscosity (NOB) RHS: face-μ Laplacian × (1/ρ̂) × Cmu + convection.
// mu/inv_rho are the property fields (physics/properties). With μ≡1, invρ≡1 it
// reduces to the const-visc RHS. (Cross-stress terms are a follow-up; they vanish
// for constant μ.)  Cmu = √(Pr/Ra) = the same nu constant.
void assemble_momentum_var_visc_cpu(const core::Domain& domain, core::CpuFields& fields,
                                    const core::CpuField& mu_field, const core::CpuField& inv_rho_field,
                                    core::CpuMomentumSystem& mom, real_t dt) {
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const real_t* up = fields[core::Var::U].data();
  const real_t* vp = fields[core::Var::V].data();
  const real_t* wp = fields[core::Var::W].data();
  const real_t  Cmu = fields.constant(core::Const::nu);
  const real_t* mu = mu_field.data(); const real_t* irho = inv_rho_field.data();
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X), *dmx1 = domain.grid.dmx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y), *dmx2 = domain.grid.dmx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z), *dmx3 = domain.grid.dmx_ptr(Direction::Z);
  // FAITHFUL per-component stage-1 residual (own×2 diffusion + skew-conv split +
  // weight-2 cross-stress, ref solvedU/W/V). Cross-stress is already folded in
  // (weight 2) → do NOT call add_transpose_stress_* on this path.
  kernels::compute_mpmstd_rhs_var(mom.rhs_u.data(), up, up, vp, wp, mu, irho,
                             dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, Cmu, dt, Direction::X);
  kernels::compute_mpmstd_rhs_var(mom.rhs_v.data(), vp, up, vp, wp, mu, irho,
                             dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, Cmu, dt, Direction::Y);
  kernels::compute_mpmstd_rhs_var(mom.rhs_w.data(), wp, up, vp, wp, mu, irho,
                             dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, Cmu, dt, Direction::Z, true);
}

// ---- one component: 3-sweep ADI (rhs_buf → out increment) -------------------
namespace {
void adi_component(core::CpuMomentumSystem& mom, std::vector<real_t>& rhs_buf,
                   const real_t* qf, const real_t* Up, const real_t* Vp, const real_t* Wp, bool z_stag,
                   const boundary::FieldBoundary& fbc, const core::Domain& domain, const boundary::BoundaryCondition& bc,
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

    kernels::build_adi_bands(dir, mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(),
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
      // Lower z-wall W face = interior row 0 of the z-sweep → identity row
      // (see the var twin below / Report/30).
      if (zs && domain.sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL) {
        for (int s = 0; s < n_sys; ++s) {
          mom.A.data()[s] = real_t{0}; mom.C.data()[s] = real_t{0};
          mom.B.data()[s] = real_t{1}; mom.D.data()[s] = real_t{0};
        }
      }
      domain.tdma.get(dir).solve_many(mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
    }
    kernels::scatter_from_tdma(dir, outb, mom.D.data(), n1, n2, n3, i1, i2, i3);
    std::swap(in, outb);
  }
  std::copy(in, in + static_cast<std::size_t>(n1) * n2 * n3, out);
}

// variable-viscosity twin (faithful per-component build_adi_bands_var)
void adi_component_var(core::CpuMomentumSystem& mom, std::vector<real_t>& rhs_buf,
                       const real_t* qf, const real_t* Up, const real_t* Vp, const real_t* Wp,
                       const real_t* mu, const real_t* irho, Direction own_dir, bool z_stag,
                       const boundary::FieldBoundary& fbc, const core::Domain& domain,
                       const boundary::BoundaryCondition& bc, real_t Cmu, real_t dt, real_t* out) {
  const auto nt = mom.n_total; const auto ni = mom.n_interior;
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int i1 = ni[0], i2 = ni[1], i3 = ni[2];
  real_t* in   = rhs_buf.data();
  real_t* outb = mom.stage.data();
  const real_t* dxo  = domain.grid.dx_ptr(own_dir);   // own-dir metric for the face interps
  const real_t* dmxo = domain.grid.dmx_ptr(own_dir);
  for (Direction dir : bc.topology.sweep_order()) {
    const real_t* dxa  = domain.grid.dx_ptr(dir);
    const real_t* dmxa = domain.grid.dmx_ptr(dir);
    const real_t* qadv = (dir == Direction::X) ? Up : (dir == Direction::Y) ? Vp : Wp;
    const bool    zs   = (dir == Direction::Z) && z_stag;
    kernels::build_adi_bands_var(dir, own_dir, mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(),
                            in, qf, qadv, mu, irho, dxa, dmxa, dxo, dmxo, n1, n2, n3, i1, i2, i3, Cmu, dt, zs);
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
      // z-staggered wall-face row: on the MAC grid the lower z-wall W face is
      // INTERIOR row 0 of the z-sweep (face zf[0], pinned δW=0). Make it an
      // identity row — a regular row lets its spurious unknown pollute the
      // adjacent rows THROUGH the TDMA during the solve, which the post-solve
      // pin cannot undo (PaScaL excludes this row via j_indexS; Report/30).
      if (zs && domain.sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL) {
        for (int s = 0; s < n_sys; ++s) {
          mom.A.data()[s] = real_t{0}; mom.C.data()[s] = real_t{0};
          mom.B.data()[s] = real_t{1}; mom.D.data()[s] = real_t{0};
        }
      }
      domain.tdma.get(dir).solve_many(mom.A.data(), mom.B.data(), mom.C.data(), mom.D.data(), n_sys, n_row);
    }
    kernels::scatter_from_tdma(dir, outb, mom.D.data(), n1, n2, n3, i1, i2, i3);
    std::swap(in, outb);
  }
  std::copy(in, in + static_cast<std::size_t>(n1) * n2 * n3, out);
}
} // anonymous namespace

// ---- solve: 3-sweep ADI per component + block coupling (M2) -----------------
void solve_momentum_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
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
  kernels::block_couple_dV(mom.dV.data(), mom.dW.data(), fields[core::Var::V].data(),
                      domain.grid.dx_ptr(Direction::Y), domain.grid.dmx_ptr(Direction::Y),
                      domain.grid.dx_ptr(Direction::Z), domain.grid.dmx_ptr(Direction::Z), n1, n2, n3, dt);
  core::sync_field_cpu(mom.dV, bc.V, domain.sub);
  kernels::block_couple_dU(mom.dU.data(), mom.dV.data(), mom.dW.data(), fields[core::Var::U].data(),
                      domain.grid.dx_ptr(Direction::X), domain.grid.dmx_ptr(Direction::X),
                      domain.grid.dmx_ptr(Direction::Y), domain.grid.dmx_ptr(Direction::Z), n1, n2, n3, dt);
}

// ---- variable-viscosity solve (NOB): 3-sweep var ADI + block coupling -------
// The lower z-wall W face lives at INTERIOR index k=kHaloWidth on the MAC grid
// (the upper wall face sits in the ghost slot), so the z-sweep solves that row
// like a regular interior row and leaves a spurious wall-face increment there.
// The face is pinned (W=0), so its increment must be zero BEFORE dW is consumed
// by block_couple_dV_var / block_couple_dU_var; otherwise it leaks into dV/dU
// with 1/dmx3 wall-shear amplification — the bottom-wall-only U spike (Report/27).
static void zero_dw_lower_wall_face(real_t* dW, const core::Domain& domain,
                                    const boundary::BoundaryCondition& bc) {
  if (bc.topology.is_periodic(Direction::Z)) return;
  if (domain.sub.topology().axis(Direction::Z).west_rank != MPI_PROC_NULL) return;
  const auto nt = domain.sub.n_total();
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2], h = kHaloWidth;
  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      dW[(i * n2 + j) * n3 + h] = real_t{0};
}

void solve_momentum_var_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                            core::CpuFields& fields, const core::CpuField& mu_field,
                            const core::CpuField& inv_rho_field,
                            core::CpuMomentumSystem& mom, real_t dt) {
  const real_t  Cmu = fields.constant(core::Const::nu);
  const real_t* Up = fields[core::Var::U].data();
  const real_t* Vp = fields[core::Var::V].data();
  const real_t* Wp = fields[core::Var::W].data();
  const real_t* mu = mu_field.data(); const real_t* irho = inv_rho_field.data();
  const real_t* dx1 = domain.grid.dx_ptr(Direction::X), *dmx1 = domain.grid.dmx_ptr(Direction::X);
  const real_t* dx2 = domain.grid.dx_ptr(Direction::Y), *dmx2 = domain.grid.dmx_ptr(Direction::Y);
  const real_t* dx3 = domain.grid.dx_ptr(Direction::Z), *dmx3 = domain.grid.dmx_ptr(Direction::Z);

  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];

  // ── FORWARD substitution dU → dV → dW (ref MPM-STD solvedU→solvedV→solvedW,
  //    submodule.f90:202-227; Report/RBC), with the lower-triangular dd*
  //    coupling injected into each RHS BEFORE its ADI solve using the freshly-
  //    solved preceding increments (Gauss-Seidel). The wall-normal dW comes
  //    LAST so every coupling into it passes through its implicit z-ADI.
  // 1) dU (own X) — no coupling.
  adi_component_var(mom, mom.rhs_u, Up, Up, Vp, Wp, mu, irho, Direction::X, false, bc.U, domain, bc, Cmu, dt, mom.dU.data());
  core::sync_field_cpu(mom.dU, bc.U, domain.sub);

  // 2) dV (own Y) — inject M21 (dU increment) into rhs_v, then solve.
  kernels::couple_dd_dV_var(mom.rhs_v.data(), mom.dU.data(), Vp, mu, irho,
                       dx1, dmx1, dx2, dmx2, n1, n2, n3, Cmu, dt);
  adi_component_var(mom, mom.rhs_v, Vp, Up, Vp, Wp, mu, irho, Direction::Y, false, bc.V, domain, bc, Cmu, dt, mom.dV.data());
  core::sync_field_cpu(mom.dV, bc.V, domain.sub);

  // 3) dW (own Z, wall) — inject M31 (dU) + M32 (dV) into rhs_w, then solve.
  kernels::couple_dd_dW_var(mom.rhs_w.data(), mom.dU.data(), mom.dV.data(), Wp, mu, irho,
                       dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, Cmu, dt);
  adi_component_var(mom, mom.rhs_w, Wp, Up, Vp, Wp, mu, irho, Direction::Z, true, bc.W, domain, bc, Cmu, dt, mom.dW.data());
  core::sync_field_cpu(mom.dW, bc.W, domain.sub);
  zero_dw_lower_wall_face(mom.dW.data(), domain, bc);

  // ── BLOCK back-substitution (upper M23/M12/M13), ref blockLdV→blockLdU order:
  // blockLdV corrects the horizontal dV using the wall-normal increment dW,
  // THEN blockLdU corrects dU using the corrected dV + dW increments. The
  // wall-normal dW receives NO post-hoc correction (Report/RBC reignition fix).
  kernels::block_couple_dV_var(mom.dV.data(), mom.dW.data(), fields[core::Var::V].data(), mu, irho,
                          dx2, dmx2, dx3, dmx3, n1, n2, n3, Cmu, dt);
  core::sync_field_cpu(mom.dV, bc.V, domain.sub);
  kernels::block_couple_dU_var(mom.dU.data(), mom.dV.data(), mom.dW.data(), fields[core::Var::U].data(), mu, irho,
                          dx1, dmx1, dx2, dmx2, dx3, dmx3, n1, n2, n3, Cmu, dt);
}

// ---- pseudo-update: q += dq (interior; caller syncs U,V,W) ------------------
void update_velocity_cpu(core::CpuFields& fields, const core::CpuMomentumSystem& mom) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  kernels::add_increment(fields[core::Var::U].data(), mom.dU.data(), n1, n2, n3);
  kernels::add_increment(fields[core::Var::V].data(), mom.dV.data(), n1, n2, n3);
  kernels::add_increment(fields[core::Var::W].data(), mom.dW.data(), n1, n2, n3);
}

} // namespace mpmstd::equation::momentum
