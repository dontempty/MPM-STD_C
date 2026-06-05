#include "equation/momentum/momentum_equation.hpp"
#include "equation/momentum/kernels/kernels.hpp"

#include <algorithm>
#include <stdexcept>

namespace mpmstd::equation::momentum {

static Direction own_dir_from_name(const std::string& name) {
  if (name == "U") return Direction::X;
  if (name == "V") return Direction::Y;
  return Direction::Z;  // "W" and any other
}

MomentumEquation::MomentumEquation(const MomentumTraits&             traits,
                                     const grid::Grid&                  grid,
                                     const parallel::mpi::Subdomain&    sub,
                                     field::FieldRegistry&              fields,
                                     const boundary::Problem&           problem,
                                     const boundary::FieldBoundary&     fbc,
                                     linear_solver::tdma::TdmaRegistry& tdma,
                                     boundary::BoundaryApplier&         bc)
  : traits_(traits),
    grid_(grid), sub_(sub), fields_(fields), problem_(problem),
    fbc_(fbc), tdma_(tdma), bc_(bc),
    own_dir_(own_dir_from_name(traits.name)) {

  if (!fields_.has_scalar(traits_.name)) {
    throw std::runtime_error(
      "MomentumEquation: field '" + traits_.name +
      "' is not registered in the FieldRegistry");
  }

  n1_tot_ = sub_.n_total()[0];   n1_int_ = sub_.n_interior()[0];
  n2_tot_ = sub_.n_total()[1];   n2_int_ = sub_.n_interior()[1];
  n3_tot_ = sub_.n_total()[2];   n3_int_ = sub_.n_interior()[2];

  const std::size_t n_full     = static_cast<std::size_t>(n1_tot_) * n2_tot_ * n3_tot_;
  const std::size_t n_interior = static_cast<std::size_t>(n1_int_) * n2_int_ * n3_int_;

  delta_    .assign(n_full,     0.0);
  stage_buf_.assign(n_full,     0.0);
  tdma_A_   .assign(n_interior, 0.0);
  tdma_B_   .assign(n_interior, 0.0);
  tdma_C_   .assign(n_interior, 0.0);
  tdma_D_   .assign(n_interior, 0.0);
}


void MomentumEquation::step(real_t dt) {
  predict(dt);
  apply_increment();
}


void MomentumEquation::predict(real_t dt) {
  const real_t* q = fields_.scalar(traits_.name).host_ptr();

  if (traits_.with_convection) {
    if (!fields_.has_scalar("U") ||
        !fields_.has_scalar("V") ||
        !fields_.has_scalar("W")) {
      throw std::runtime_error(
        "MomentumEquation: with_convection=true but U/V/W are not "
        "registered in the FieldRegistry.");
    }
  }

  // -------------------------------------------------------------------
  // (1) Explicit RHS = full physical residual:
  //     delta = dt*nu*(Lx+Ly+Lz)(q^n) - dt*div(u*q^n)
  //
  //   conv_f=1.0 so the converged balance is nu*lap(u)=div(u*q) (physical).
  //   This equals MPM-STD's NET explicit residual (its 0.25 split + the
  //   -M11*u^n / -M12*v^n / -M13*w^n delta-form terms sum to the same thing).
  //   With conv_f=0.25 (the bare MPM-STD band coefficient, without the
  //   -M11*u^n correction this code does not carry) the convection would be
  //   4x too weak — no nonlinear energy transfer, the channel relaminarizes.
  //   The 0.25*dt implicit convection in build_adi_bands acts on the increment
  //   (vanishes at steady state) and only stabilizes the advection sweep.
  //   conv_f=0.0 when with_convection=false.
  // -------------------------------------------------------------------
  const bool   z_stag = (traits_.name == "W");
  const real_t conv_f = traits_.with_convection ? real_t{1.0} : real_t{0};

  // Velocity pointers for convection RHS. If no convection, U/V/W
  // are not used (conv_f=0 multiplies the flux to zero).
  const real_t* U_ptr = (traits_.with_convection && fields_.has_scalar("U"))
                        ? fields_.scalar("U").host_ptr() : q;
  const real_t* V_ptr = (traits_.with_convection && fields_.has_scalar("V"))
                        ? fields_.scalar("V").host_ptr() : q;
  const real_t* W_ptr = (traits_.with_convection && fields_.has_scalar("W"))
                        ? fields_.scalar("W").host_ptr() : q;

  kernels::compute_mpmstd_rhs(
      delta_.data(), q,
      U_ptr, V_ptr, W_ptr,
      grid_.dx_ptr(Direction::X), grid_.dmx_ptr(Direction::X),
      grid_.dx_ptr(Direction::Y), grid_.dmx_ptr(Direction::Y),
      grid_.dx_ptr(Direction::Z), grid_.dmx_ptr(Direction::Z),
      n1_tot_, n2_tot_, n3_tot_,
      traits_.viscosity, dt,
      own_dir_,
      z_stag,
      conv_f);

  // (1c) Source term (buoyancy / body force): delta += dt * source
  if (!traits_.source_name.empty()) {
    if (!fields_.has_scalar(traits_.source_name)) {
      throw std::runtime_error(
        "MomentumEquation: source field '" + traits_.source_name +
        "' is not registered in the FieldRegistry.");
    }
    kernels::add_source_rhs(
        delta_.data(),
        fields_.scalar(traits_.source_name).host_ptr(),
        n1_tot_, n2_tot_, n3_tot_,
        dt);
  }

  // (1d) Constant body force (e.g. mean pressure gradient for channel flow)
  if (traits_.constant_source != real_t{0}) {
    kernels::add_constant_rhs(delta_.data(), n1_tot_, n2_tot_, n3_tot_,
                               dt * traits_.constant_source);
  }

  // -------------------------------------------------------------------
  // (2) Three ADI stages (ping-pong between delta_ and stage_buf_).
  //     Own direction: visc_factor=1.0 (backward Euler).
  //     Cross directions: visc_factor=0.5 (Crank-Nicolson).
  // -------------------------------------------------------------------
  const auto order = problem_.topology.sweep_order();

  real_t* in_buf  = delta_.data();
  real_t* out_buf = stage_buf_.data();

  for (Direction d : order) {
    const bool is_own = (d == own_dir_);
    adi_stage_(d, dt, in_buf, out_buf, is_own);
    std::swap(in_buf, out_buf);
  }
  // After 3 swaps, in_buf holds the final δ — copy back to delta_ if needed.
  if (in_buf != delta_.data()) {
    std::copy(in_buf, in_buf + delta_.size(), delta_.data());
  }
}


void MomentumEquation::apply_increment() {
  field::ScalarField& q_field = fields_.scalar(traits_.name);
  real_t* q = q_field.host_ptr();

  kernels::add_increment(q, delta_.data(), n1_tot_, n2_tot_, n3_tot_);

  q_field.exchange_halo();
  bc_.apply_ghost(q_field, fbc_);
}


void MomentumEquation::adi_stage_(Direction d, real_t dt,
                                    const real_t* src, real_t* dst,
                                    bool is_own) {
  const real_t* dx_along  = grid_.dx_ptr (d);
  const real_t* dmx_along = grid_.dmx_ptr(d);

  // W face-centered in z → z-ADI needs the face-centered stencil.
  const bool z_stag = (d == Direction::Z) && (traits_.name == "W");

  // Crank-Nicolson on the viscous term: visc_factor = 0.5 for ALL directions.
  //
  // For CONSTANT viscosity this is the correct specialization of the MPM-STD
  // Beam-Warming scheme.  MPM-STD doubles the own-direction implicit coefficient
  // (×2 → full) but only as part of a delta form that also subtracts M11·u^n
  // (the implicit operator applied to the current field) AND adds the variable-
  // viscosity cross-stress terms; for constant viscosity those combine to a net
  // Crank-Nicolson treatment.  Our explicit RHS is dt·nu·(Lx+Ly+Lz)u^n (no own
  // doubling — it cancels against the cross-stress via continuity), so the
  // consistent implicit half is 0.5·dt·nu on every direction.
  //
  // Using 1.0 on the own direction WITHOUT the −M11·u^n delta-form correction
  // makes the wall-normal sweep backward-Euler (amplification 1/(1+dt·λ) instead
  // of the CN (1−0.5dt·λ)/(1+0.5dt·λ)), over-damping near-wall fluctuations and
  // suppressing sub-critical bypass transition.
  (void)is_own;
  const real_t visc_factor = real_t{0.5};

  // Advecting velocity for convective band terms.
  // Direction X → advect by U;  Y → V;  Z → W.
  const real_t* q_field = fields_.scalar(traits_.name).host_ptr();
  const real_t* q_adv;
  if (d == Direction::X) {
    q_adv = fields_.has_scalar("U") ? fields_.scalar("U").host_ptr() : q_field;
  } else if (d == Direction::Y) {
    q_adv = fields_.has_scalar("V") ? fields_.scalar("V").host_ptr() : q_field;
  } else {
    q_adv = fields_.has_scalar("W") ? fields_.scalar("W").host_ptr() : q_field;
  }

  // If no convection, zero out convective bands by passing q_adv=q_field
  // with cconv=0.  The band builder uses 0.25*dt multiplied by q_adv face
  // velocities; since with_convection=false means q_adv plays no role we
  // still pass it but the coefficient is zeroed implicitly via no-conv path.
  // (Actually the band builder always uses 0.25*dt; the caller must
  //  ensure q_adv ≈ 0 or that convective bands are desired.)
  // For with_convection=false, use the field itself so face velocities
  // cancel when subtracted (self-advection → d/dx(q*q)/2 which is O(q²),
  // typically negligible, but ideally conv_factor in band builder = 0).
  // TODO: add conv_factor to build_adi_bands.  For now channel always
  // has with_convection=true so this path is unused.

  kernels::build_adi_bands(d,
                            tdma_A_.data(), tdma_B_.data(),
                            tdma_C_.data(), tdma_D_.data(),
                            src,
                            q_field,
                            q_adv,
                            dx_along, dmx_along,
                            n1_tot_, n2_tot_, n3_tot_,
                            n1_int_, n2_int_, n3_int_,
                            traits_.viscosity, dt,
                            visc_factor,
                            z_stag);

  int n_row = 0, n_sys = 0;
  switch (d) {
    case Direction::X: n_row = n1_int_; n_sys = n2_int_ * n3_int_; break;
    case Direction::Y: n_row = n2_int_; n_sys = n1_int_ * n3_int_; break;
    case Direction::Z: n_row = n3_int_; n_sys = n1_int_ * n2_int_; break;
  }

  if (problem_.topology.is_periodic(d)) {
    tdma_.get(d).solve_many_cyclic(
        tdma_A_.data(), tdma_B_.data(), tdma_C_.data(), tdma_D_.data(),
        n_sys, n_row);
  } else {
    const auto& axis_comm = sub_.topology().axis(d);
    bc_.modify_tdma_row(d, fbc_, axis_comm,
                          tdma_A_.data(), tdma_B_.data(),
                          tdma_C_.data(), tdma_D_.data(),
                          n_sys, n_row);
    tdma_.get(d).solve_many(
        tdma_A_.data(), tdma_B_.data(), tdma_C_.data(), tdma_D_.data(),
        n_sys, n_row);
  }

  kernels::scatter_from_tdma(d, dst, tdma_D_.data(),
                              n1_tot_, n2_tot_, n3_tot_,
                              n1_int_, n2_int_, n3_int_);
}

} // namespace mpmstd::equation::momentum
