#pragma once

#include "equation/scalar/scalar_traits.hpp"
#include "boundary/boundary_applier.hpp"
#include "boundary/problem.hpp"
#include "field/field_registry.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <vector>

namespace mpmstd::equation::scalar {

// ScalarEquation — advances one passive scalar (or temperature) by `dt`
// using Crank-Nicolson + Douglas-style 3-stage ADI for the diffusion operator.
//
// Phase 1 (now): pure diffusion ∂φ/∂t = α ∇²φ
//                periodic axes only (cyclic TDMA on every axis).
// Phase 2 :     Dirichlet BC on a wall axis (modify_tdma_row).
// Phase 3 :     convection -u·∇φ added to the explicit RHS.
//
// Storage:
//   * The scalar's host buffer lives in FieldRegistry under `traits_.name`.
//   * Per-stage workspace (delta field, TDMA bands) is owned by this class.

class ScalarEquation {
public:
  // `fbc` is the per-scalar FieldBoundary (e.g. problem.T for the thermal
  // solver, problem.<some_other> for a passive-species solver).  Storing it
  // explicitly keeps ScalarEquation agnostic of any particular Problem field.
  ScalarEquation(const ScalarTraits& traits,
                  const grid::Grid& grid,
                  const parallel::mpi::Subdomain& sub,
                  field::FieldRegistry& fields,
                  const boundary::Problem& problem,
                  const boundary::FieldBoundary& fbc,
                  linear_solver::tdma::TdmaRegistry& tdma,
                  boundary::BoundaryApplier& bc);

  ScalarEquation(const ScalarEquation&) = delete;
  ScalarEquation& operator=(const ScalarEquation&) = delete;

  // Advance the scalar by `dt`.
  void step(real_t dt);

  const ScalarTraits& traits() const { return traits_; }

private:
  // Internal helpers (one per stage).
  void compute_explicit_rhs_(real_t* rhs, const real_t* T, real_t dt) const;
  void adi_stage_           (Direction d, real_t dt,
                              const real_t* src, real_t* dst);

  // Owning workspace buffers (resized at construction time).
  std::vector<real_t> delta_;             // n_total^3
  std::vector<real_t> stage_buf_;         // n_total^3 (intermediate)
  std::vector<real_t> tdma_A_, tdma_B_, tdma_C_, tdma_D_;   // interior (n_int^3)

  // References (stored as raw refs — these outlive the equation).
  ScalarTraits                          traits_;
  const grid::Grid&                     grid_;
  const parallel::mpi::Subdomain&       sub_;
  field::FieldRegistry&                 fields_;
  const boundary::Problem&              problem_;
  const boundary::FieldBoundary&        fbc_;       // per-scalar BC table
  linear_solver::tdma::TdmaRegistry&    tdma_;
  boundary::BoundaryApplier&            bc_;

  // Cached extents.
  int n1_tot_, n2_tot_, n3_tot_;
  int n1_int_, n2_int_, n3_int_;
};

} // namespace mpmstd::equation::scalar
