#pragma once

// rev.2 P1: pressure Poisson solver ported to CpuField — copied verbatim from
// the validated PressureSolver and adapted at the 3 field touchpoints
// (divergence read, dP unpack, projection). The heavy FFT/transpose/TDMA
// internals are field-agnostic and unchanged. Decoupled from the old
// FieldRegistry / ScalarField-based BoundaryApplier (BC via core::apply_ghost_cpu).

#include "common/types.hpp"
#include "core/field_cpu.hpp"
#include "core/grid.hpp"                          // grid::Grid
#include "core/boundary.hpp"                      // boundary::Problem
#include "core/mpi_topology.hpp"                  // Subdomain
#include "core/halo.hpp"                          // exchange_halo_cpu
#include "core/boundary_ops.hpp"                  // apply_ghost_cpu
#include "linear_solver/tdma/tdma_registry.hpp"

namespace mpmstd::equation::pressure {

// Abstract base: common divergence RHS / unpack / projection (identical across
// FFT and DCT solvers). Concrete solvers override the spectral + TDMA steps.
class PressureSolverBase {
public:
  virtual ~PressureSolverBase() = default;

  virtual void solve(real_t dt,
                     core::CpuField& U, core::CpuField& V, core::CpuField& W, core::CpuField& P) = 0;

  PressureSolverBase(const PressureSolverBase&)            = delete;
  PressureSolverBase& operator=(const PressureSolverBase&) = delete;

protected:
  PressureSolverBase(const grid::Grid&                  grid,
                     const parallel::mpi::Subdomain&    sub,
                     const boundary::Problem&           problem,
                     linear_solver::tdma::TdmaRegistry& tdma);

  void compute_divergence_rhs_(real_t* buf,
                               const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                               real_t dt) const;

  void unpack_from_buf_(const real_t* buf, real_t scale, core::CpuField& P);

  void project_(real_t dt,
                core::CpuField& U, core::CpuField& V, core::CpuField& W, const core::CpuField& P);

  // ── Common state ─────────────────────────────────────────────────────────
  const grid::Grid&                  grid_;
  const parallel::mpi::Subdomain&    sub_;
  const boundary::Problem&           problem_;
  linear_solver::tdma::TdmaRegistry& tdma_;

  int n1_tot_, n2_tot_, n3_tot_;
  int n1_int_, n2_int_, n3_int_;
};

} // namespace mpmstd::equation::pressure
