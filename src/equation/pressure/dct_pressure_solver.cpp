#include "equation/pressure/dct_pressure_solver.hpp"
#include "common/macros.hpp"

#include <fftw3.h>
#include <mpi.h>
#include <cmath>
#include <stdexcept>

namespace mpmstd::equation::pressure {

namespace {

// Interior-only buffer index: [k * n1 * n2 + i * n2 + j].
inline int idx3f(int k, int i, int j, int n1, int n2) {
  return (k * n1 + i) * n2 + j;
}

} // anonymous namespace


// ── constructor / destructor ─────────────────────────────────────────────────

DctPressureSolver::DctPressureSolver(const grid::Grid&                  grid,
                                     const parallel::mpi::Subdomain&    sub,
                                     field::FieldRegistry&              fields,
                                     const boundary::Problem&           problem,
                                     linear_solver::tdma::TdmaRegistry& tdma,
                                     boundary::BoundaryApplier&         bc)
  : PressureSolverBase(grid, sub, fields, problem, tdma, bc) {

  if (problem_.topology.is_periodic(Direction::X) ||
      problem_.topology.is_periodic(Direction::Y)) {
    throw std::runtime_error(
      "DctPressureSolver: both X and Y must be non-periodic. "
      "Mixed periodic/Neumann is not yet supported.");
  }
  if (sub_.topology().axis(Direction::X).nprocs > 1 ||
      sub_.topology().axis(Direction::Y).nprocs > 1) {
    throw std::runtime_error(
      "DctPressureSolver: X/Y MPI decomposition not supported (M4).");
  }

  // --- Allocate buffers ---
  const std::size_t n_int   = static_cast<std::size_t>(n1_int_) * n2_int_ * n3_int_;
  const std::size_t n_bands = static_cast<std::size_t>(n3_int_) * n1_int_ * n2_int_;

  dct_buf_.resize(n_int,   0.0);
  A_.resize(n_bands, 0.0);  B_.resize(n_bands, 0.0);
  C_.resize(n_bands, 0.0);  D_.resize(n_bands, 0.0);

  // --- FFTW r2r plans (in-place, 2D, batched over z) ---
  const int            n_r2r[2]   = { n1_int_, n2_int_ };
  const fftw_r2r_kind  fwd_k[2]   = { FFTW_REDFT10, FFTW_REDFT10 };
  const fftw_r2r_kind  bwd_k[2]   = { FFTW_REDFT01, FFTW_REDFT01 };

  plan_fwd_ = reinterpret_cast<fftw_plan_s*>(
    fftw_plan_many_r2r(
      2, n_r2r, n3_int_,
      dct_buf_.data(), nullptr, 1, n1_int_ * n2_int_,
      dct_buf_.data(), nullptr, 1, n1_int_ * n2_int_,
      fwd_k, FFTW_ESTIMATE));

  plan_bwd_ = reinterpret_cast<fftw_plan_s*>(
    fftw_plan_many_r2r(
      2, n_r2r, n3_int_,
      dct_buf_.data(), nullptr, 1, n1_int_ * n2_int_,
      dct_buf_.data(), nullptr, 1, n1_int_ * n2_int_,
      bwd_k, FFTW_ESTIMATE));

  // --- Eigenvalues ---
  // λ_x[mx] = (1 - cos(π*mx/N1)) * K_x   (DCT-II / Neumann eigenvalue)
  // K_x = average FD stiffness from actual grid metrics.
  // For uniform dx: K_x = 2/dx² exactly.
  const int h    = kHaloWidth;
  const int N1_g = sub_.n_global()[0];
  const int N2_g = sub_.n_global()[1];

  const real_t* dx1  = grid_.dx_ptr (Direction::X);
  const real_t* dmx1 = grid_.dmx_ptr(Direction::X);
  const real_t* dx2  = grid_.dx_ptr (Direction::Y);
  const real_t* dmx2 = grid_.dmx_ptr(Direction::Y);

  double sum_x = 0.0;
  for (int ii = 0; ii < n1_int_; ++ii) {
    const int i = ii + h;
    sum_x += 1.0 / (dx1[i] * dmx1[i]) + 1.0 / (dx1[i] * dmx1[i + 1]);
  }
  const double K_x = sum_x / n1_int_;

  double sum_y = 0.0;
  for (int jj = 0; jj < n2_int_; ++jj) {
    const int j = jj + h;
    sum_y += 1.0 / (dx2[j] * dmx2[j]) + 1.0 / (dx2[j] * dmx2[j + 1]);
  }
  const double K_y = sum_y / n2_int_;

  lambda_x_.resize(n1_int_);
  for (int mx = 0; mx < n1_int_; ++mx) {
    const double arg = M_PI * mx / N1_g;
    lambda_x_[mx] = static_cast<real_t>((1.0 - std::cos(arg)) * K_x);
  }

  lambda_y_.resize(n2_int_);
  for (int my = 0; my < n2_int_; ++my) {
    const double arg = M_PI * my / N2_g;
    lambda_y_[my] = static_cast<real_t>((1.0 - std::cos(arg)) * K_y);
  }
}


DctPressureSolver::~DctPressureSolver() {
  if (plan_fwd_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_fwd_));
  if (plan_bwd_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_bwd_));
}


// ── public interface ─────────────────────────────────────────────────────────

void DctPressureSolver::solve(real_t dt,
                               field::ScalarField& U,
                               field::ScalarField& V,
                               field::ScalarField& W,
                               field::ScalarField& P) {
  compute_divergence_rhs_(dct_buf_.data(), U, V, W, dt);
  forward_dct_xy_();
  solve_tdma_z_();
  inverse_dct_xy_();
  const real_t scale = static_cast<real_t>(1.0) / (4.0 * n1_int_ * n2_int_);
  unpack_from_buf_(dct_buf_.data(), scale, P);
  project_(dt, U, V, W, P);
}


// ── private sub-steps ────────────────────────────────────────────────────────

void DctPressureSolver::forward_dct_xy_() {
  fftw_execute(reinterpret_cast<fftw_plan>(plan_fwd_));
}


void DctPressureSolver::solve_tdma_z_() {
  const real_t* dx3  = grid_.dx_ptr (Direction::Z);
  const real_t* dmx3 = grid_.dmx_ptr(Direction::Z);
  const int h = kHaloWidth;

  const int n_sys = n1_int_ * n2_int_;
  const int n_row = n3_int_;

  const bool owns_lower = (sub_.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
  const bool owns_upper = (sub_.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);

  for (int kk = 0; kk < n3_int_; ++kk) {
    const int    k     = kk + h;
    const real_t a_val = static_cast<real_t>(1.0) / (dx3[k] * dmx3[k  ]);
    const real_t c_val = static_cast<real_t>(1.0) / (dx3[k] * dmx3[k+1]);
    const real_t Lz    = -(a_val + c_val);

    for (int mx = 0; mx < n1_int_; ++mx) {
      for (int my = 0; my < n2_int_; ++my) {
        const int    sys   = mx * n2_int_ + my;
        const int    p     = kk * n_sys + sys;
        const real_t lam   = lambda_x_[mx] + lambda_y_[my];

        A_[p] = a_val;
        B_[p] = Lz - lam;
        C_[p] = c_val;
        D_[p] = dct_buf_[idx3f(kk, mx, my, n1_int_, n2_int_)];
      }
    }
  }

  if (owns_lower) {
    for (int sys = 0; sys < n_sys; ++sys) {
      B_[sys] += A_[sys];  A_[sys] = 0.0;
    }
  }
  if (owns_upper) {
    const int last = (n_row - 1) * n_sys;
    for (int sys = 0; sys < n_sys; ++sys) {
      B_[last + sys] += C_[last + sys];  C_[last + sys] = 0.0;
    }
  }

  const bool pin_mean = owns_lower
                        && (sub_.global_offset(Direction::X) == 0)
                        && (sub_.global_offset(Direction::Y) == 0);
  if (pin_mean) {
    A_[0] = 0.0;  B_[0] = 1.0;  C_[0] = 0.0;  D_[0] = 0.0;
  }

  const bool periodic_z = problem_.topology.is_periodic(Direction::Z);
  if (periodic_z) {
    tdma_.get(Direction::Z).solve_many_cyclic(
        A_.data(), B_.data(), C_.data(), D_.data(), n_sys, n_row);
  } else {
    tdma_.get(Direction::Z).solve_many(
        A_.data(), B_.data(), C_.data(), D_.data(), n_sys, n_row);
  }

  for (int kk = 0; kk < n3_int_; ++kk)
    for (int mx = 0; mx < n1_int_; ++mx)
      for (int my = 0; my < n2_int_; ++my)
        dct_buf_[idx3f(kk, mx, my, n1_int_, n2_int_)] =
            D_[kk * n_sys + mx * n2_int_ + my];
}


void DctPressureSolver::inverse_dct_xy_() {
  fftw_execute(reinterpret_cast<fftw_plan>(plan_bwd_));
}

} // namespace mpmstd::equation::pressure
