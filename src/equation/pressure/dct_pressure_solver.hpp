#pragma once

#include "equation/pressure/pressure_solver_base.hpp"

#include <vector>

struct fftw_plan_s;

namespace mpmstd::equation::pressure {

// DctPressureSolver — DCT-based pressure Poisson solver for cavity / duct flows.
//
// Requirement: X and Y must both be non-periodic (Neumann pressure walls).
//              Z may be periodic or wall (handled by TDMA).
//
// Eigenvalues: λ_x[m] = (1-cos(π*m/N1)) * K_x   (DCT-II / Neumann)
//   K_x = average FD stiffness from actual grid metrics (handles non-uniform X).
//   For uniform grid K_x = 2/dx² exactly.
//
// Normalization: REDFT10 ∘ REDFT01 = 2N per dimension → scale = 1/(4*N1*N2).
class DctPressureSolver : public PressureSolverBase {
public:
  DctPressureSolver(const grid::Grid&                  grid,
                    const parallel::mpi::Subdomain&    sub,
                    field::FieldRegistry&              fields,
                    const boundary::Problem&           problem,
                    linear_solver::tdma::TdmaRegistry& tdma,
                    boundary::BoundaryApplier&         bc);
  ~DctPressureSolver();

  DctPressureSolver(const DctPressureSolver&)            = delete;
  DctPressureSolver& operator=(const DctPressureSolver&) = delete;

  void solve(real_t dt,
             field::ScalarField& U,
             field::ScalarField& V,
             field::ScalarField& W,
             field::ScalarField& P) override;

private:
  void forward_dct_xy_();
  void solve_tdma_z_();
  void inverse_dct_xy_();

  // Real DCT buffer (in-place): [k * n1_int * n2_int + i * n2_int + j]
  std::vector<real_t> dct_buf_;

  fftw_plan_s* plan_fwd_ = nullptr;   // REDFT10 (DCT-II) forward
  fftw_plan_s* plan_bwd_ = nullptr;   // REDFT01 (DCT-III) inverse

  // λ_x[mx] = (1-cos(π*mx/N1)) * K_x,  λ_y[my] = (1-cos(π*my/N2)) * K_y
  std::vector<real_t> lambda_x_;
  std::vector<real_t> lambda_y_;

  // TDMA bands: [kk * n_sys + (mx * n2_int + my)],  n_sys = n1_int * n2_int
  std::vector<real_t> A_, B_, C_, D_;
};

} // namespace mpmstd::equation::pressure
