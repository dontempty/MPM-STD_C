#pragma once

#include "equation/momentum/momentum_traits.hpp"
#include "boundary/boundary_applier.hpp"
#include "boundary/problem.hpp"
#include "field/field_registry.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <vector>

namespace mpmstd::equation::momentum {

// MomentumEquation — advances one velocity component q (U, V, or W) by dt
// using Crank-Nicolson + Douglas-style 3-stage ADI for the viscous term.
//
//     ∂q/∂t + u·∇q = ν ∇²q + source
//
// Algorithm (per step):
//   (1) Explicit RHS:  δ = dt·ν·(Lx+Ly+Lz) q^n − dt·u·∇q^n + dt·source
//   (2) 3 ADI stages:  (I − 0.5·dt·ν·L_d) δ^(k+1) = δ^(k)  (Douglas form)
//   (3) q^{n+1} = q^n + δ
//   (4) Halo exchange + apply_ghost
//
// Source term (M3-B+):
//   If traits.source_name is non-empty, the named field is read from the
//   FieldRegistry and added as dt * source_field to the explicit RHS.
//   Caller is responsible for filling the source field before each step().
//
// Convection (optional):
//   When traits.with_convection == true, the velocity fields "U", "V", "W"
//   must be registered in the FieldRegistry.  Their halos must be valid
//   before step() is called.
//
// Note on stagger (M3-A):
//   The storage shape of q is n_total^3 regardless of stagger convention.
//   The wall BC treatment (ghost-cell placement for face-centred fields)
//   and face-interpolated advection velocities are deferred to M3-B when
//   non-periodic axes are introduced.

class MomentumEquation {
public:
  // `fbc` is the per-component FieldBoundary (e.g. problem.U for the
  //  x-momentum equation, problem.V for y-momentum, etc.).
  MomentumEquation(const MomentumTraits&             traits,
                    const grid::Grid&                  grid,
                    const parallel::mpi::Subdomain&    sub,
                    field::FieldRegistry&              fields,
                    const boundary::Problem&           problem,
                    const boundary::FieldBoundary&     fbc,
                    linear_solver::tdma::TdmaRegistry& tdma,
                    boundary::BoundaryApplier&         bc);

  MomentumEquation(const MomentumEquation&) = delete;
  MomentumEquation& operator=(const MomentumEquation&) = delete;

  // One-shot advance: predict → apply_increment → halo.
  void step(real_t dt);

  // Two-phase interface for NOB/block_couple use cases:
  //
  //   predict(dt)        — compute δ, store in delta_ (does NOT modify q).
  //   apply_increment()  — q += δ, then halo + apply_ghost.
  //
  // Between predict() and apply_increment(), the caller may read delta_ via
  // delta_ptr() to apply block_couple corrections before committing.
  void predict(real_t dt);
  void apply_increment();

  // Pointer to the internal increment buffer (valid after predict(), reset
  // to zero by the next predict() call).
  const real_t* delta_ptr() const { return delta_.data(); }
  real_t*       delta_ptr()       { return delta_.data(); }

  const MomentumTraits& traits() const { return traits_; }

private:
  // is_own: true when d is the stagger direction of q (U→X, V→Y, W→Z).
  // Uses full dt*nu coefficient (backward Euler) instead of 0.5*dt*nu (CN).
  void adi_stage_(Direction d, real_t dt,
                  const real_t* src, real_t* dst, bool is_own);

  // Workspace
  std::vector<real_t> delta_;
  std::vector<real_t> stage_buf_;
  std::vector<real_t> tdma_A_, tdma_B_, tdma_C_, tdma_D_;

  // Non-owning references
  MomentumTraits                        traits_;
  const grid::Grid&                     grid_;
  const parallel::mpi::Subdomain&       sub_;
  field::FieldRegistry&                 fields_;
  const boundary::Problem&              problem_;
  const boundary::FieldBoundary&        fbc_;
  linear_solver::tdma::TdmaRegistry&    tdma_;
  boundary::BoundaryApplier&            bc_;

  // Own direction: the stagger axis of this velocity component.
  // U → X,  V → Y,  W → Z.
  Direction own_dir_;

  int n1_tot_, n2_tot_, n3_tot_;
  int n1_int_, n2_int_, n3_int_;
};

} // namespace mpmstd::equation::momentum
