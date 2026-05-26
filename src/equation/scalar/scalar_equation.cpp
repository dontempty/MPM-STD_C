#include "equation/scalar/scalar_equation.hpp"
#include "equation/scalar/kernels/kernels.hpp"

#include <stdexcept>

namespace mpmstd::equation::scalar {

ScalarEquation::ScalarEquation(const ScalarTraits& traits,
                                 const grid::Grid& grid,
                                 const parallel::mpi::Subdomain& sub,
                                 field::FieldRegistry& fields,
                                 const boundary::Problem& problem,
                                 const boundary::FieldBoundary& fbc,
                                 linear_solver::tdma::TdmaRegistry& tdma,
                                 boundary::BoundaryApplier& bc)
  : traits_(traits),
    grid_(grid), sub_(sub), fields_(fields), problem_(problem),
    fbc_(fbc), tdma_(tdma), bc_(bc) {

  // The scalar must already be registered.
  if (!fields_.has_scalar(traits_.name)) {
    throw std::runtime_error(
      "ScalarEquation: scalar '" + traits_.name +
      "' is not registered in the FieldRegistry");
  }

  n1_tot_ = sub_.n_total()[0];   n1_int_ = sub_.n_interior()[0];
  n2_tot_ = sub_.n_total()[1];   n2_int_ = sub_.n_interior()[1];
  n3_tot_ = sub_.n_total()[2];   n3_int_ = sub_.n_interior()[2];

  // Workspace: two halo'd buffers (delta_, stage_buf_) and four TDMA bands
  // sized to the global interior cell count.
  const std::size_t n_full     = static_cast<std::size_t>(n1_tot_) * n2_tot_ * n3_tot_;
  const std::size_t n_interior = static_cast<std::size_t>(n1_int_) * n2_int_ * n3_int_;

  delta_    .assign(n_full,    0.0);
  stage_buf_.assign(n_full,    0.0);
  tdma_A_   .assign(n_interior, 0.0);
  tdma_B_   .assign(n_interior, 0.0);
  tdma_C_   .assign(n_interior, 0.0);
  tdma_D_   .assign(n_interior, 0.0);
}


void ScalarEquation::step(real_t dt) {
  field::ScalarField& T_field = fields_.scalar(traits_.name);
  real_t* T = T_field.host_ptr();

  // -----------------------------------------------------------------------
  // (1) Explicit RHS:  rhs = dt * alpha * (Lx + Ly + Lz) T^n  −  dt * u·∇T^n
  //
  //     Halos (interior + global) must be set BEFORE this call.  The
  //     caller's time loop guarantees that — at the end of step() we
  //     halo-exchange + apply_ghost the field.
  // -----------------------------------------------------------------------
  kernels::laplacian_explicit_rhs(
      delta_.data(), T,
      grid_.dx_ptr(Direction::X), grid_.dmx_ptr(Direction::X),
      grid_.dx_ptr(Direction::Y), grid_.dmx_ptr(Direction::Y),
      grid_.dx_ptr(Direction::Z), grid_.dmx_ptr(Direction::Z),
      n1_tot_, n2_tot_, n3_tot_,
      dt * traits_.diffusivity);

  // (1b) Convection — only if requested AND U/V/W are registered.
  if (traits_.with_convection) {
    if (!fields_.has_scalar("U") ||
        !fields_.has_scalar("V") ||
        !fields_.has_scalar("W")) {
      throw std::runtime_error(
        "ScalarEquation: with_convection=true but the velocity fields "
        "U / V / W are not registered in the FieldRegistry.");
    }
    kernels::add_convection_rhs(
        delta_.data(), T,
        fields_.scalar("U").host_ptr(),
        fields_.scalar("V").host_ptr(),
        fields_.scalar("W").host_ptr(),
        grid_.dx_ptr(Direction::X),
        grid_.dx_ptr(Direction::Y),
        grid_.dx_ptr(Direction::Z),
        n1_tot_, n2_tot_, n3_tot_,
        -dt);
  }

  // -----------------------------------------------------------------------
  // (2) Three ADI stages, in the order dictated by the problem topology.
  //     Stages 2 and 3 use the previous stage's output as their input.
  //     We ping-pong between `delta_` and `stage_buf_`.
  // -----------------------------------------------------------------------
  const auto order = problem_.topology.sweep_order();

  real_t* in_buf  = delta_.data();
  real_t* out_buf = stage_buf_.data();

  for (Direction d : order) {
    adi_stage_(d, dt, in_buf, out_buf);
    std::swap(in_buf, out_buf);
  }
  // After 3 swaps, `in_buf` holds the final increment δ.

  // -----------------------------------------------------------------------
  // (3) T^{n+1} = T^n + δ
  // -----------------------------------------------------------------------
  kernels::add_increment(T, in_buf, n1_tot_, n2_tot_, n3_tot_);

  // -----------------------------------------------------------------------
  // (4) Halo + global-boundary update so that the next stencil call has
  //     well-defined ghosts.
  // -----------------------------------------------------------------------
  T_field.exchange_halo();
  bc_.apply_ghost(T_field, fbc_);
}


void ScalarEquation::adi_stage_(Direction d, real_t dt,
                                  const real_t* src, real_t* dst) {
  // Pick the on-axis grid metrics.
  const real_t* dx_along  = grid_.dx_ptr (d);
  const real_t* dmx_along = grid_.dmx_ptr(d);

  // Build the per-axis bands + pack the RHS into TDMA layout.
  kernels::build_adi_bands(d,
                            tdma_A_.data(), tdma_B_.data(), tdma_C_.data(), tdma_D_.data(),
                            src,
                            dx_along, dmx_along,
                            n1_tot_, n2_tot_, n3_tot_,
                            n1_int_, n2_int_, n3_int_,
                            traits_.diffusivity, dt);

  // System count + row count depend on the axis.
  int n_row = 0, n_sys = 0;
  switch (d) {
    case Direction::X: n_row = n1_int_; n_sys = n2_int_ * n3_int_; break;
    case Direction::Y: n_row = n2_int_; n_sys = n1_int_ * n3_int_; break;
    case Direction::Z: n_row = n3_int_; n_sys = n1_int_ * n2_int_; break;
  }

  if (problem_.topology.is_periodic(d)) {
    // Cyclic TDMA — no wall row modification.
    tdma_.get(d).solve_many_cyclic(
        tdma_A_.data(), tdma_B_.data(), tdma_C_.data(), tdma_D_.data(),
        n_sys, n_row);
  } else {
    // Non-cyclic TDMA: enforce the wall BC by amending row 0 / n_row-1 on
    // the ranks that own the lower / upper global wall.
    const auto& axis_comm = sub_.topology().axis(d);
    bc_.modify_tdma_row(d, fbc_, axis_comm,
                          tdma_A_.data(), tdma_B_.data(), tdma_C_.data(), tdma_D_.data(),
                          n_sys, n_row);
    tdma_.get(d).solve_many(
        tdma_A_.data(), tdma_B_.data(), tdma_C_.data(), tdma_D_.data(),
        n_sys, n_row);
  }

  // Scatter the solution back to `dst` (interior cells only).
  kernels::scatter_from_tdma(d, dst, tdma_D_.data(),
                              n1_tot_, n2_tot_, n3_tot_,
                              n1_int_, n2_int_, n3_int_);
}


// Phase-1: compute_explicit_rhs_ is folded into the body of step() — kept here
// as a forward-compatible hook for Phase 3 when convection enters the RHS.
void ScalarEquation::compute_explicit_rhs_(real_t* /*rhs*/,
                                             const real_t* /*T*/,
                                             real_t /*dt*/) const {
  // intentionally empty in Phase 1
}

} // namespace mpmstd::equation::scalar
