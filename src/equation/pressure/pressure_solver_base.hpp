#pragma once

#include "boundary/boundary_applier.hpp"
#include "boundary/problem.hpp"
#include "common/types.hpp"
#include "field/field_registry.hpp"
#include "field/scalar_field.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "parallel/mpi/subdomain.hpp"

namespace mpmstd::equation::pressure {

// Abstract base for pressure Poisson solvers.
//
// Holds the common state shared by all concrete solvers (grid, BC, TDMA,
// sizes) and provides three protected methods that are identical across
// all implementations:
//
//   compute_divergence_rhs_(buf, U, V, W, dt)
//       Fills buf[k*n1*n2 + i*n2 + j] = div(U*)[i,j,k] / dt.
//
//   unpack_from_buf_(buf, scale, P)
//       Writes buf*scale → P interior, removes global mean, applies halo + ghost.
//       Uses double-precision MPI accumulation regardless of real_t.
//
//   project_(dt, U, V, W, P)
//       U -= dt * grad(P), halo-exchange, apply ghost.
//
// Concrete implementations override only the spectral + TDMA steps:
//   PressureSolver     — 2D R2C FFT  (periodic X, Y)
//   DctPressureSolver  — 2D DCT-II   (Neumann  X, Y)
class PressureSolverBase {
public:
  virtual ~PressureSolverBase() = default;

  virtual void solve(real_t dt,
                     field::ScalarField& U,
                     field::ScalarField& V,
                     field::ScalarField& W,
                     field::ScalarField& P) = 0;

  PressureSolverBase(const PressureSolverBase&)            = delete;
  PressureSolverBase& operator=(const PressureSolverBase&) = delete;

protected:
  PressureSolverBase(const grid::Grid&                  grid,
                     const parallel::mpi::Subdomain&    sub,
                     field::FieldRegistry&              fields,
                     const boundary::Problem&           problem,
                     linear_solver::tdma::TdmaRegistry& tdma,
                     boundary::BoundaryApplier&         bc);

  void compute_divergence_rhs_(real_t*                   buf,
                                const field::ScalarField& U,
                                const field::ScalarField& V,
                                const field::ScalarField& W,
                                real_t                    dt) const;

  // scale = 1/(transform_normalization).  Uses double accumulation for MPI.
  void unpack_from_buf_(const real_t* buf, real_t scale, field::ScalarField& P);

  void project_(real_t dt,
                field::ScalarField& U,
                field::ScalarField& V,
                field::ScalarField& W,
                const field::ScalarField& P);

  // ── Common state ─────────────────────────────────────────────────────────
  const grid::Grid&                  grid_;
  const parallel::mpi::Subdomain&    sub_;
  field::FieldRegistry&              fields_;
  const boundary::Problem&           problem_;
  linear_solver::tdma::TdmaRegistry& tdma_;
  boundary::BoundaryApplier&         bc_;

  int n1_tot_, n2_tot_, n3_tot_;
  int n1_int_, n2_int_, n3_int_;
};

} // namespace mpmstd::equation::pressure
