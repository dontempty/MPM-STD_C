#include "equation/pressure/pressure_engine.hpp"
#include "common/macros.hpp"

#include <fftw3.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mpmstd::equation::pressure {

namespace {

// ---------------------------------------------------------------------------
// Flat index helpers (0-based interior, no halo)
//   C-layout: x-fastest, k-slowest   idx_C(i,j,k) = i + nx*(j + ny*k)
//   I-layout: x-fastest, k-slowest   idx_I(i,j,k) = i + n1m*(j + ny*k)
//   Y-layout: ix-fastest, k-slowest  idx_Y(ix,j,k) = ix + h1pY*(j + n2m*k)
// ---------------------------------------------------------------------------
inline std::size_t idx_C(int i, int j, int k, int nx, int ny)
{ return (std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * k); }

inline std::size_t idx_I(int i, int j, int k, int n1m, int ny)
{ return (std::size_t)i + (std::size_t)n1m * ((std::size_t)j + (std::size_t)ny * k); }

inline std::size_t idx_Y(int ix, int j, int k, int h1pY, int n2m)
{ return (std::size_t)ix + (std::size_t)h1pY * ((std::size_t)j + (std::size_t)n2m * k); }

} // anonymous namespace


// ── constructor ──────────────────────────────────────────────────────────────

PressureSolver::PressureSolver(const grid::Grid&                  grid,
                                const parallel::mpi::Subdomain&    sub,
                                const boundary::Problem&           problem,
                                linear_solver::tdma::TdmaRegistry& tdma)
  : PressureSolverBase(grid, sub, problem, tdma)
{
  if (!problem_.topology.is_periodic(Direction::X) ||
      !problem_.topology.is_periodic(Direction::Y)) {
    throw std::runtime_error(
      "PressureSolver: X and Y must be periodic — use DctPressureSolver for wall BCs.");
  }

  // 1. Local sizes
  nx_loc_ = n1_int_;   // sub_.n_interior()[0]
  ny_loc_ = n2_int_;   // sub_.n_interior()[1]
  nz_loc_ = n3_int_;   // sub_.n_interior()[2]

  const int N1g = sub_.n_global()[0];
  const int N2g = sub_.n_global()[1];
  const int N3g = sub_.n_global()[2];
  Nxh_ = N1g / 2 + 1;

  // 2. I-pencil: z is further split by np1 (the same pattern as PaScaL_TCS FFT1)
  const int np1 = sub_.topology().dims()[0];
  const int np2 = sub_.topology().dims()[1];
  const int np3 = sub_.topology().dims()[2];
  (void)np3;  // used implicitly via nprocs

  if (nz_loc_ % np1 != 0)
    throw std::runtime_error(
      "PressureSolver: nz_loc not divisible by np1 — adjust grid or process count.");
  n3_I_ = nz_loc_ / np1;

  // PaScaL_TDMA's distributed algorithm requires each rank to own at least 3
  // z-rows of the tridiagonal system (the forward/backward sweeps are needed to
  // eliminate interior rows before packing boundary rows into the reduced system).
  // n3_I_ must be >= 3 for the comm_xz distributed solve to be correct.
  // For single-rank comm_xz (np1*np3 == 1), n3_I_ == nz_loc_ and PaScaL uses
  // its local solver, which imposes no lower bound.
  const int size_xz_check = sub_.topology().size_xz();
  if (size_xz_check > 1 && n3_I_ < 3)
    throw std::runtime_error(
      "PressureSolver: n3_I (= nz_loc/np1 = " + std::to_string(n3_I_) +
      ") must be >= 3 when comm_xz has more than one rank (nprocs_xz = " +
      std::to_string(size_xz_check) + "). "
      "Increase the grid size or reduce np1*np3.");

  // 3. Y-pencil: x-freq split by np2 (para_range style)
  h1p_Y_.resize(np2);
  ix_start_Y_.resize(np2);
  {
    int base = Nxh_ / np2, extra = Nxh_ % np2;
    for (int r = 0; r < np2; ++r) {
      h1p_Y_[r]      = base + (r < extra ? 1 : 0);
      ix_start_Y_[r] = r * base + std::min(r, extra);
    }
  }
  const int ry      = sub_.topology().axis(Direction::Y).rank;
  h1p_Y_me_      = h1p_Y_[ry];
  ix_start_Y_me_ = ix_start_Y_[ry];
  n_sys_Y_       = h1p_Y_me_ * N2g;   // systems for the TDMA z solve

  // 4. Wavenumbers
  //    lambda_x: half-spectrum [0, Nxh_), modified wavenumber using grid metric K_x
  //    lambda_y: full spectrum [0, N2g_), using K_y
  {
    const int h = kHaloWidth;
    const real_t* dx1  = grid_.dx_ptr (Direction::X);
    const real_t* dmx1 = grid_.dmx_ptr(Direction::X);
    const real_t* dx2  = grid_.dx_ptr (Direction::Y);
    const real_t* dmx2 = grid_.dmx_ptr(Direction::Y);

    // K_x: average FD stiffness from local grid (all ranks share same x-metrics
    // because X decomposition only splits the interior cells)
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

    lambda_x_.resize(Nxh_);
    for (int mx = 0; mx < Nxh_; ++mx) {
      const double arg = 2.0 * M_PI * mx / N1g;
      lambda_x_[mx] = (1.0 - std::cos(arg)) * K_x;
    }

    lambda_y_.resize(N2g);
    for (int j = 0; j < N2g; ++j) {
      const int    jj  = (j <= N2g / 2) ? j : j - N2g;
      const double arg = 2.0 * M_PI * jj / N2g;
      lambda_y_[j] = (1.0 - std::cos(arg)) * K_y;
    }
  }

  // 5. Global z-metrics via Allgather over axis(Z).comm
  //    dz_g_[k_g+1]  = cell width      at global interior cell k_g (0-based)
  //    dmz_g_[k_g+1] = face half-width at lower face of cell k_g
  //    dmz_g_[k_g+2] = face half-width at upper face of cell k_g
  {
    const int h = kHaloWidth;
    const real_t* dx3_local  = grid_.dx_ptr (Direction::Z);
    const real_t* dmx3_local = grid_.dmx_ptr(Direction::Z);

    dz_g_ .assign(N3g + 2, 0.0);
    dmz_g_.assign(N3g + 2, 0.0);

    // Each Z-rank contributes nz_loc_ values.
    // Gather cell widths: interior indices h..h+nz_loc_-1 → global [1..N3g]
    if (sizeof(real_t) == sizeof(double)) {
      MPI_Allgather(dx3_local  + h, nz_loc_, MPI_DOUBLE,
                    dz_g_.data()  + 1, nz_loc_, MPI_DOUBLE,
                    sub_.topology().axis(Direction::Z).comm);
      MPI_Allgather(dmx3_local + h, nz_loc_, MPI_DOUBLE,
                    dmz_g_.data() + 1, nz_loc_, MPI_DOUBLE,
                    sub_.topology().axis(Direction::Z).comm);
    } else {
      // real_t is float: gather into a temporary double buffer then copy
      std::vector<float> tmp_dx3(nz_loc_), tmp_dmx3(nz_loc_);
      for (int k = 0; k < nz_loc_; ++k) {
        tmp_dx3[k]  = static_cast<float>(dx3_local [h + k]);
        tmp_dmx3[k] = static_cast<float>(dmx3_local[h + k]);
      }
      std::vector<float> gbuf_dx3(N3g), gbuf_dmx3(N3g);
      MPI_Allgather(tmp_dx3.data(),  nz_loc_, MPI_FLOAT,
                    gbuf_dx3.data(),  nz_loc_, MPI_FLOAT,
                    sub_.topology().axis(Direction::Z).comm);
      MPI_Allgather(tmp_dmx3.data(), nz_loc_, MPI_FLOAT,
                    gbuf_dmx3.data(), nz_loc_, MPI_FLOAT,
                    sub_.topology().axis(Direction::Z).comm);
      for (int k = 0; k < N3g; ++k) {
        dz_g_ [k + 1] = static_cast<double>(gbuf_dx3 [k]);
        dmz_g_[k + 1] = static_cast<double>(gbuf_dmx3[k]);
      }
    }

    // Upper wall face distance: on the last Z-rank, dmx3[h+nz_loc_] is the
    // upper-wall face spacing.  Broadcast from last rank to all.
    double dmz_last = static_cast<double>(dmx3_local[h + nz_loc_]);
    MPI_Bcast(&dmz_last, 1, MPI_DOUBLE,
              sub_.topology().axis(Direction::Z).nprocs - 1,
              sub_.topology().axis(Direction::Z).comm);
    dmz_g_[N3g + 1] = dmz_last;
  }

  // 6. Allocate pencil buffers
  rhs_C_.assign((std::size_t)nx_loc_ * ny_loc_ * nz_loc_, 0.0);
  rhs_I_.assign((std::size_t)N1g     * ny_loc_ * n3_I_,   0.0);
  hat_I_.assign((std::size_t)Nxh_    * ny_loc_ * n3_I_,   {0.0, 0.0});
  hat_Y_.assign((std::size_t)h1p_Y_me_ * N2g   * n3_I_,   {0.0, 0.0});
  dp_I_ .assign((std::size_t)N1g     * ny_loc_ * n3_I_,   0.0);
  dp_C_ .assign((std::size_t)nx_loc_ * ny_loc_ * nz_loc_, real_t{0});

  // 7. Pre-allocate transpose buffers
  {
    const std::size_t blk_C = (std::size_t)nx_loc_ * ny_loc_ * n3_I_;
    tx_sbuf_C_.resize(np1 * blk_C);
    tx_rbuf_C_.resize(np1 * blk_C);

    // I↔Y: the forward direction sends Nxh*yn3*2 total doubles,
    // the backward direction sends np2*(h1p_Y_me_*yn3)*2.  Take max.
    const std::size_t yn3   = (std::size_t)ny_loc_ * n3_I_;
    const std::size_t nfwd  = (std::size_t)Nxh_          * yn3 * 2;
    const std::size_t nbwd  = (std::size_t)np2 * (std::size_t)h1p_Y_me_ * yn3 * 2;
    const std::size_t n_cplx_Y = std::max(nfwd, nbwd);
    tx_sbuf_Y_.resize(n_cplx_Y);
    tx_rbuf_Y_.resize(n_cplx_Y);
  }

  // 8. FFTW plans (1D batched, matching Filtered_TDMA PressureSolver.cpp)
  {
    // x: batched r2c of length N1g, howmany = ny_loc_*n3_I_
    int nx[1] = { N1g };
    plan_fwd_x_ = reinterpret_cast<fftw_plan_s*>(
      fftw_plan_many_dft_r2c(
        1, nx, ny_loc_ * n3_I_,
        rhs_I_.data(), nullptr, 1, N1g,
        reinterpret_cast<fftw_complex*>(hat_I_.data()), nullptr, 1, Nxh_,
        FFTW_ESTIMATE));
    plan_bwd_x_ = reinterpret_cast<fftw_plan_s*>(
      fftw_plan_many_dft_c2r(
        1, nx, ny_loc_ * n3_I_,
        reinterpret_cast<fftw_complex*>(hat_I_.data()), nullptr, 1, Nxh_,
        dp_I_.data(), nullptr, 1, N1g,
        FFTW_ESTIMATE));
  }
  if (h1p_Y_me_ > 0 && n3_I_ > 0) {
    // y: batched c2c of length N2g, howmany = h1p_Y_me_
    // hat_Y_[ix + h1p*(j + N2g*k)]: stride = h1p_Y_me_, dist = 1
    int ny[1] = { N2g };
    auto* p = reinterpret_cast<fftw_complex*>(hat_Y_.data());
    plan_fwd_y_ = reinterpret_cast<fftw_plan_s*>(
      fftw_plan_many_dft(
        1, ny, h1p_Y_me_,
        p, nullptr, h1p_Y_me_, 1,
        p, nullptr, h1p_Y_me_, 1,
        FFTW_FORWARD, FFTW_ESTIMATE));
    plan_bwd_y_ = reinterpret_cast<fftw_plan_s*>(
      fftw_plan_many_dft(
        1, ny, h1p_Y_me_,
        p, nullptr, h1p_Y_me_, 1,
        p, nullptr, h1p_Y_me_, 1,
        FFTW_BACKWARD, FFTW_ESTIMATE));
  }

  // 9. Distributed z-TDMA using comm_xz
  {
    const int rank_xz = sub_.topology().rank_xz();
    const int size_xz = sub_.topology().size_xz();
    comm_xz_1d_.comm      = sub_.topology().comm_xz();
    comm_xz_1d_.rank      = rank_xz;
    comm_xz_1d_.nprocs    = size_xz;
    comm_xz_1d_.west_rank = (rank_xz > 0)           ? rank_xz - 1 : MPI_PROC_NULL;
    comm_xz_1d_.east_rank = (rank_xz < size_xz - 1) ? rank_xz + 1 : MPI_PROC_NULL;

    tdma_z_ = std::make_unique<linear_solver::tdma::PaScaLTDMACpuBackend>(comm_xz_1d_);

    const std::size_t tsz = (std::size_t)n3_I_ * std::max(n_sys_Y_, 1);
    tdma_A_r_.resize(tsz);  tdma_B_r_.resize(tsz);
    tdma_C_r_.resize(tsz);  tdma_D_r_.resize(tsz);
    tdma_A_c_.resize(tsz);  tdma_B_c_.resize(tsz);
    tdma_C_c_.resize(tsz);  tdma_D_c_.resize(tsz);
  }
}


// ── destructor ───────────────────────────────────────────────────────────────

PressureSolver::~PressureSolver()
{
  if (plan_fwd_x_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_fwd_x_));
  if (plan_bwd_x_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_bwd_x_));
  if (plan_fwd_y_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_fwd_y_));
  if (plan_bwd_y_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_bwd_y_));
}


// ── public interface ─────────────────────────────────────────────────────────

void PressureSolver::solve(real_t dt,
                            core::CpuField& U,
                            core::CpuField& V,
                            core::CpuField& W,
                            core::CpuField& P)
{
  // Cache current-solve arguments so helper methods can access them.
  cur_U_  = &U;
  cur_V_  = &V;
  cur_W_  = &W;
  cur_dt_ = dt;

  // 1. RHS: divergence / dt into rhs_C_ (C-layout, x-fastest)
  compute_divergence_rhs_pencil_();

  // 2. Forward: C→I, FFT(x), I→Y, FFT(y)
  transpose_C_to_I_();
  fft_x_forward_();
  transpose_I_to_Y_();
  fft_y_forward_();

  // 3. Distributed z-TDMA over comm_xz
  solve_tdma_z_();

  // 4. Backward: IFFT(y), Y→I, IFFT(x)
  fft_y_backward_();
  transpose_Y_to_I_();
  fft_x_backward_();

  // 5. Scale (FFTW is un-normalised) then transpose I→C to fill dp_C_
  const int N1g = sub_.n_global()[0];
  const int N2g = sub_.n_global()[1];
  const double scale = 1.0 / (static_cast<double>(N1g) * static_cast<double>(N2g));
  for (std::size_t q = 0; q < dp_I_.size(); ++q) dp_I_[q] *= scale;

  std::fill(dp_C_.begin(), dp_C_.end(), real_t{0});
  transpose_I_to_C_();   // fills dp_C_ (C-layout, x-fastest, no halo)

  // 6. Unpack dp_C_ into P field interior, remove global mean, apply ghost+halo.
  //    This matches the original unpack_from_buf_ semantics: P is overwritten with
  //    dP (not accumulated).  The base-class project_() then uses P (= dP) to
  //    correct U, V, W.
  {
    const int h   = kHaloWidth;
    const int n2t = sub_.n_total()[1];
    const int n3t = sub_.n_total()[2];

    // Remove global mean (for uniqueness; same as base-class unpack_from_buf_).
    double local_sum = 0.0;
    const std::size_t n_loc = (std::size_t)nx_loc_ * ny_loc_ * nz_loc_;
    for (std::size_t q = 0; q < n_loc; ++q)
      local_sum += static_cast<double>(dp_C_[q]);

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  sub_.topology().cart_comm());
    const int    n_int_global = sub_.n_global()[0] * sub_.n_global()[1] * sub_.n_global()[2];
    const real_t mean_dp = static_cast<real_t>(global_sum / n_int_global);

    // Write dP - mean into P interior (MPM-STD_C field layout: x-slowest, z-fastest).
    real_t* p = P.data();
    for (int kk = 0; kk < nz_loc_; ++kk)
      for (int jj = 0; jj < ny_loc_; ++jj)
        for (int ii = 0; ii < nx_loc_; ++ii) {
          const int i = ii + h, j = jj + h, k = kk + h;
          p[(i * n2t + j) * n3t + k] =
              dp_C_[idx_C(ii, jj, kk, nx_loc_, ny_loc_)] - mean_dp;
        }

    // Apply Neumann ghost for dP at Z walls (Neumann BC: ghost = first interior).
    const int rz      = sub_.topology().axis(Direction::Z).rank;
    const int np3_val = sub_.topology().axis(Direction::Z).nprocs;
    if (rz == 0) {
      for (int jj = 0; jj < ny_loc_; ++jj)
        for (int ii = 0; ii < nx_loc_; ++ii) {
          const int i = ii + h, j = jj + h;
          p[(i * n2t + j) * n3t + (h - 1)] = p[(i * n2t + j) * n3t + h];
        }
    }
    if (rz == np3_val - 1) {
      for (int jj = 0; jj < ny_loc_; ++jj)
        for (int ii = 0; ii < nx_loc_; ++ii) {
          const int i = ii + h, j = jj + h;
          p[(i * n2t + j) * n3t + (h + nz_loc_)] =
              p[(i * n2t + j) * n3t + (h + nz_loc_ - 1)];
        }
    }

    // Halo exchange for dP (x, y are periodic; bc_.apply_ghost handles Z walls).
    core::exchange_halo_cpu(P, sub_);
    core::apply_ghost_cpu(P, problem_.P, sub_);
  }

  // 7. Project: U,V,W -= dt * grad(P),   halo + ghost applied inside project_().
  project_(dt, U, V, W, P);
}


// ── private sub-steps ────────────────────────────────────────────────────────

// ---------------------------------------------------------------------------
// RHS: (1/dt) div(u*) in C-layout (x-fastest)
// MPM-STD_C field layout: f[(i*n2t + j)*n3t + k]  (x-slowest, z-fastest)
// ---------------------------------------------------------------------------
void PressureSolver::compute_divergence_rhs_pencil_()
{
  const real_t* u = cur_U_->data();
  const real_t* v = cur_V_->data();
  const real_t* w = cur_W_->data();

  const real_t* dx1 = grid_.dx_ptr(Direction::X);
  const real_t* dx2 = grid_.dx_ptr(Direction::Y);
  const real_t* dx3 = grid_.dx_ptr(Direction::Z);

  const int h   = kHaloWidth;
  const int n2t = sub_.n_total()[1];
  const int n3t = sub_.n_total()[2];
  const double inv_dt = 1.0 / static_cast<double>(cur_dt_);

  for (int kk = 0; kk < nz_loc_; ++kk) {
    const int k  = kk + h;
    const int kp = k + 1;
    for (int jj = 0; jj < ny_loc_; ++jj) {
      const int j  = jj + h;
      const int jp = j + 1;
      for (int ii = 0; ii < nx_loc_; ++ii) {
        const int i  = ii + h;
        const int ip = i + 1;

        const double div =
            (u[(ip * n2t + j) * n3t + k ] - u[(i * n2t + j) * n3t + k ]) / dx1[i]
          + (v[(i * n2t + jp) * n3t + k ] - v[(i * n2t + j) * n3t + k ]) / dx2[j]
          + (w[(i * n2t + j ) * n3t + kp] - w[(i * n2t + j) * n3t + k ]) / dx3[k];

        rhs_C_[idx_C(ii, jj, kk, nx_loc_, ny_loc_)] = div * inv_dt;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// C → I  (MPI_Alltoall on axis(X).comm, np1 ranks)
// rhs_C_[nx_loc × ny_loc × nz_loc] → rhs_I_[n1m × ny_loc × n3_I]
// ---------------------------------------------------------------------------
void PressureSolver::transpose_C_to_I_()
{
  const int np1 = sub_.topology().dims()[0];
  const int blk = nx_loc_ * ny_loc_ * n3_I_;
  double* sbuf = tx_sbuf_C_.data();
  double* rbuf = tx_rbuf_C_.data();

  // Pack: send to rank s the z-rows [s*n3_I_, (s+1)*n3_I_)
  for (int s = 0; s < np1; ++s) {
    double* sb = sbuf + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          sb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k] =
              rhs_C_[idx_C(i, j, s * n3_I_ + k, nx_loc_, ny_loc_)];
  }

  MPI_Alltoall(sbuf, blk, MPI_DOUBLE,
               rbuf, blk, MPI_DOUBLE,
               sub_.topology().axis(Direction::X).comm);

  // Unpack: from rank s → x-block [s*nx_loc_, (s+1)*nx_loc_)
  const int N1g = sub_.n_global()[0];
  for (int s = 0; s < np1; ++s) {
    const double* rb = rbuf + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          rhs_I_[idx_I(s * nx_loc_ + i, j, k, N1g, ny_loc_)] =
              rb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k];
  }
}

// ---------------------------------------------------------------------------
// I → C  (reverse of C→I)
// dp_I_[n1m × ny_loc × n3_I] → dp_C_[nx_loc × ny_loc × nz_loc] (C-layout)
// ---------------------------------------------------------------------------
void PressureSolver::transpose_I_to_C_()
{
  const int np1 = sub_.topology().dims()[0];
  const int blk = nx_loc_ * ny_loc_ * n3_I_;
  double* sbuf = tx_sbuf_C_.data();
  double* rbuf = tx_rbuf_C_.data();
  const int N1g = sub_.n_global()[0];

  // Pack: from rank s x-block [s*nx_loc_]
  for (int s = 0; s < np1; ++s) {
    double* sb = sbuf + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          sb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k] =
              dp_I_[idx_I(s * nx_loc_ + i, j, k, N1g, ny_loc_)];
  }

  MPI_Alltoall(sbuf, blk, MPI_DOUBLE,
               rbuf, blk, MPI_DOUBLE,
               sub_.topology().axis(Direction::X).comm);

  // Unpack: to z-rows [s*n3_I_, (s+1)*n3_I_)
  for (int s = 0; s < np1; ++s) {
    const double* rb = rbuf + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          dp_C_[idx_C(i, j, s * n3_I_ + k, nx_loc_, ny_loc_)] =
              static_cast<real_t>(rb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k]);
  }
}

// ---------------------------------------------------------------------------
// I → Y  (MPI_Alltoallv on axis(Y).comm, np2 ranks)
// hat_I_[Nxh × ny_loc × n3_I] → hat_Y_[h1p_Y_me × n2m × n3_I]
// ---------------------------------------------------------------------------
void PressureSolver::transpose_I_to_Y_()
{
  const int np2 = sub_.topology().dims()[1];
  const int yn3 = ny_loc_ * n3_I_;
  const int N2g = sub_.n_global()[1];

  std::vector<int> scnts(np2), sdsp(np2), rcnts(np2), rdsp(np2);
  {
    int off = 0;
    for (int r = 0; r < np2; ++r) {
      scnts[r] = h1p_Y_[r] * yn3 * 2;  // *2: complex → 2 doubles
      sdsp[r]  = off;
      off     += scnts[r];
    }
  }
  const int rcnt_each = h1p_Y_me_ * yn3 * 2;
  for (int r = 0; r < np2; ++r) {
    rcnts[r] = rcnt_each;
    rdsp[r]  = r * rcnt_each;
  }

  double* sbuf = tx_sbuf_Y_.data();
  double* rbuf = tx_rbuf_Y_.data();

  // Pack: to rank r, send x-modes [ix_start_Y_[r]..+h1p_Y_[r]-1], all j, all k
  {
    int off = 0;
    for (int r = 0; r < np2; ++r) {
      double* sb  = sbuf + off;
      int     ix0 = ix_start_Y_[r], nxr = h1p_Y_[r];
      for (int ixl = 0; ixl < nxr; ++ixl)
        for (int j = 0; j < ny_loc_; ++j)
          for (int k = 0; k < n3_I_; ++k) {
            auto c = hat_I_[idx_I(ix0 + ixl, j, k, Nxh_, ny_loc_)];
            int  p = ((std::size_t)ixl * yn3 + j * n3_I_ + k) * 2;
            sb[p] = c.real();  sb[p + 1] = c.imag();
          }
      off += scnts[r];
    }
  }

  MPI_Alltoallv(sbuf, scnts.data(), sdsp.data(), MPI_DOUBLE,
                rbuf, rcnts.data(), rdsp.data(), MPI_DOUBLE,
                sub_.topology().axis(Direction::Y).comm);

  // Unpack: from rank r → y-block [r*ny_loc_, (r+1)*ny_loc_)
  for (int r = 0; r < np2; ++r) {
    const double* rb    = rbuf + (std::size_t)r * rcnt_each;
    const int     j_base = r * ny_loc_;
    for (int ixl = 0; ixl < h1p_Y_me_; ++ixl)
      for (int jl = 0; jl < ny_loc_; ++jl)
        for (int k = 0; k < n3_I_; ++k) {
          int p = ((std::size_t)ixl * yn3 + jl * n3_I_ + k) * 2;
          hat_Y_[idx_Y(ixl, j_base + jl, k, h1p_Y_me_, N2g)] = {rb[p], rb[p + 1]};
        }
  }
}

// ---------------------------------------------------------------------------
// Y → I  (reverse of I→Y)
// ---------------------------------------------------------------------------
void PressureSolver::transpose_Y_to_I_()
{
  const int np2 = sub_.topology().dims()[1];
  const int yn3 = ny_loc_ * n3_I_;
  const int N2g = sub_.n_global()[1];

  const int scnt_each = h1p_Y_me_ * yn3 * 2;

  std::vector<int> scnts(np2), sdsp(np2), rcnts(np2), rdsp(np2);
  for (int r = 0; r < np2; ++r) {
    scnts[r] = scnt_each;
    sdsp[r]  = r * scnt_each;
    rcnts[r] = h1p_Y_[r] * yn3 * 2;
  }
  {
    int off = 0;
    for (int r = 0; r < np2; ++r) { rdsp[r] = off; off += rcnts[r]; }
  }

  double* sbuf = tx_sbuf_Y_.data();
  double* rbuf = tx_rbuf_Y_.data();

  // Pack: to rank r, send y-block [r*ny_loc_, (r+1)*ny_loc_)
  for (int r = 0; r < np2; ++r) {
    double*   sb     = sbuf + (std::size_t)r * scnt_each;
    const int j_base = r * ny_loc_;
    for (int ixl = 0; ixl < h1p_Y_me_; ++ixl)
      for (int jl = 0; jl < ny_loc_; ++jl)
        for (int k = 0; k < n3_I_; ++k) {
          auto c = hat_Y_[idx_Y(ixl, j_base + jl, k, h1p_Y_me_, N2g)];
          int  p = ((std::size_t)ixl * yn3 + jl * n3_I_ + k) * 2;
          sb[p] = c.real();  sb[p + 1] = c.imag();
        }
  }

  MPI_Alltoallv(sbuf, scnts.data(), sdsp.data(), MPI_DOUBLE,
                rbuf, rcnts.data(), rdsp.data(), MPI_DOUBLE,
                sub_.topology().axis(Direction::Y).comm);

  // Unpack: from rank r → x-modes [ix_start_Y_[r]..+h1p_Y_[r]-1]
  int off = 0;
  for (int r = 0; r < np2; ++r) {
    const double* rb  = rbuf + off;
    int           ix0 = ix_start_Y_[r], nxr = h1p_Y_[r];
    for (int ixl = 0; ixl < nxr; ++ixl)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k) {
          int p = ((std::size_t)ixl * yn3 + j * n3_I_ + k) * 2;
          hat_I_[idx_I(ix0 + ixl, j, k, Nxh_, ny_loc_)] = {rb[p], rb[p + 1]};
        }
    off += rcnts[r];
  }
}

// ---------------------------------------------------------------------------
// FFT wrappers
// ---------------------------------------------------------------------------
void PressureSolver::fft_x_forward_()
{
  if (plan_fwd_x_) fftw_execute(reinterpret_cast<fftw_plan>(plan_fwd_x_));
}

void PressureSolver::fft_x_backward_()
{
  if (plan_bwd_x_) fftw_execute(reinterpret_cast<fftw_plan>(plan_bwd_x_));
}

void PressureSolver::fft_y_forward_()
{
  if (!plan_fwd_y_) return;
  auto* p = reinterpret_cast<fftw_complex*>(hat_Y_.data());
  const int N2g = sub_.n_global()[1];
  const std::size_t stride = (std::size_t)h1p_Y_me_ * N2g;
  for (int k = 0; k < n3_I_; ++k)
    fftw_execute_dft(reinterpret_cast<fftw_plan>(plan_fwd_y_),
                     p + k * stride, p + k * stride);
}

void PressureSolver::fft_y_backward_()
{
  if (!plan_bwd_y_) return;
  auto* p = reinterpret_cast<fftw_complex*>(hat_Y_.data());
  const int N2g = sub_.n_global()[1];
  const std::size_t stride = (std::size_t)h1p_Y_me_ * N2g;
  for (int k = 0; k < n3_I_; ++k)
    fftw_execute_dft(reinterpret_cast<fftw_plan>(plan_bwd_y_),
                     p + k * stride, p + k * stride);
}

// ---------------------------------------------------------------------------
// solve_tdma_z_
// Distributed non-cyclic TDMA over comm_xz (np1 * np3 ranks).
// Y-pencil: hat_Y_[h1p_Y_me_ × n2m × n3_I_]
// System index: s = ixl + h1p_Y_me_ * j  (0-based)
// k_global (0-based) = rank_xz * n3_I_ + k
// ---------------------------------------------------------------------------
void PressureSolver::solve_tdma_z_()
{
  if (!tdma_z_ || n_sys_Y_ == 0 || n3_I_ == 0) return;

  const int N2g     = sub_.n_global()[1];
  const int rank_xz = sub_.topology().rank_xz();
  const int size_xz = sub_.topology().size_xz();

  for (int k = 0; k < n3_I_; ++k) {
    const int    k_g  = rank_xz * n3_I_ + k;   // 0-based global z-index
    const double dz   = dz_g_ [k_g + 1];
    const double dmz  = dmz_g_[k_g + 1];
    const double dmzp = dmz_g_[k_g + 2];

    const bool lower_wall = (rank_xz == 0            && k == 0);
    const bool upper_wall = (rank_xz == size_xz - 1  && k == n3_I_ - 1);

    const double a_val = lower_wall ? 0.0 : 1.0 / (dz * dmz );
    const double c_val = upper_wall ? 0.0 : 1.0 / (dz * dmzp);

    for (int j = 0; j < N2g; ++j)
      for (int ixl = 0; ixl < h1p_Y_me_; ++ixl) {
        const int         s   = ixl + h1p_Y_me_ * j;
        const std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
        const double lam      = lambda_x_[ix_start_Y_me_ + ixl] + lambda_y_[j];

        tdma_A_r_[idx] = tdma_A_c_[idx] = a_val;
        tdma_C_r_[idx] = tdma_C_c_[idx] = c_val;
        tdma_B_r_[idx] = tdma_B_c_[idx] = -(a_val + c_val) - lam;

        const auto cv = hat_Y_[idx_Y(ixl, j, k, h1p_Y_me_, N2g)];
        tdma_D_r_[idx] = cv.real();
        tdma_D_c_[idx] = cv.imag();
      }
  }

  // Pin (0,0) mode: set to zero on rank_xz == 0
  if (rank_xz == 0 && ix_start_Y_me_ == 0 && h1p_Y_me_ > 0) {
    tdma_A_r_[0] = tdma_A_c_[0] = 0.0;
    tdma_C_r_[0] = tdma_C_c_[0] = 0.0;
    tdma_B_r_[0] = tdma_B_c_[0] = 1.0;
    tdma_D_r_[0] = tdma_D_c_[0] = 0.0;
  }

  // Solve: non-cyclic (wall boundary in z)
  tdma_z_->solve_many(tdma_A_r_.data(), tdma_B_r_.data(), tdma_C_r_.data(), tdma_D_r_.data(),
                       n_sys_Y_, n3_I_);
  tdma_z_->solve_many(tdma_A_c_.data(), tdma_B_c_.data(), tdma_C_c_.data(), tdma_D_c_.data(),
                       n_sys_Y_, n3_I_);

  // Write solution back to hat_Y_
  for (int k = 0; k < n3_I_; ++k)
    for (int j = 0; j < N2g; ++j)
      for (int ixl = 0; ixl < h1p_Y_me_; ++ixl) {
        const int         s   = ixl + h1p_Y_me_ * j;
        const std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
        hat_Y_[idx_Y(ixl, j, k, h1p_Y_me_, N2g)] = {tdma_D_r_[idx], tdma_D_c_[idx]};
      }
}

} // namespace mpmstd::equation::pressure
