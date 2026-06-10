#include "equation/pressure/pressure_base.hpp"
#include "common/macros.hpp"

#include <mpi.h>
#include <cmath>

namespace mpmstd::equation::pressure {

namespace {

inline int idx3h(int i, int j, int k, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

// Interior-only buffer layout: [k * n1 * n2 + i * n2 + j]  (k-slowest, j-fastest).
inline int idx3f(int k, int i, int j, int n1, int n2) {
  return (k * n1 + i) * n2 + j;
}

} // anonymous namespace


// ── constructor ──────────────────────────────────────────────────────────────

PressureSolverBase::PressureSolverBase(const grid::Grid&                  grid,
                                       const parallel::mpi::Subdomain&    sub,
                                       const boundary::Problem&           problem,
                                       linear_solver::tdma::TdmaRegistry& tdma)
  : grid_(grid), sub_(sub), problem_(problem), tdma_(tdma) {

  n1_tot_ = sub_.n_total()[0];   n1_int_ = sub_.n_interior()[0];
  n2_tot_ = sub_.n_total()[1];   n2_int_ = sub_.n_interior()[1];
  n3_tot_ = sub_.n_total()[2];   n3_int_ = sub_.n_interior()[2];
}


// ── compute_divergence_rhs_ ──────────────────────────────────────────────────

void PressureSolverBase::compute_divergence_rhs_(real_t*                   buf,
                                                  const core::CpuField& U,
                                                  const core::CpuField& V,
                                                  const core::CpuField& W,
                                                  real_t                    dt) const {
  const real_t* u = U.data();
  const real_t* v = V.data();
  const real_t* w = W.data();

  const real_t* dx1 = grid_.dx_ptr(Direction::X);
  const real_t* dx2 = grid_.dx_ptr(Direction::Y);
  const real_t* dx3 = grid_.dx_ptr(Direction::Z);

  const int n2 = n2_tot_, n3 = n3_tot_;
  const int h  = kHaloWidth;

  for (int ii = 0; ii < n1_int_; ++ii) {
    const int i = ii + h, ip = i + 1;
    for (int jj = 0; jj < n2_int_; ++jj) {
      const int j = jj + h, jp = j + 1;
      for (int kk = 0; kk < n3_int_; ++kk) {
        const int k = kk + h, kp = k + 1;

        const real_t div_u =
            (u[idx3h(ip, j,  k,  n2, n3)] - u[idx3h(i, j, k, n2, n3)]) / dx1[i]
          + (v[idx3h(i,  jp, k,  n2, n3)] - v[idx3h(i, j, k, n2, n3)]) / dx2[j]
          + (w[idx3h(i,  j,  kp, n2, n3)] - w[idx3h(i, j, k, n2, n3)]) / dx3[k];

        buf[idx3f(kk, ii, jj, n1_int_, n2_int_)] = div_u / dt;
      }
    }
  }
}


// ── unpack_from_buf_ ─────────────────────────────────────────────────────────

void PressureSolverBase::unpack_from_buf_(const real_t* buf,
                                           real_t        scale,
                                           core::CpuField& P) {
  const int h = kHaloWidth;

  // Accumulate in double for precision; MPI_DOUBLE is always correct here.
  double local_sum = 0.0;
  const std::size_t n_int = static_cast<std::size_t>(n1_int_) * n2_int_ * n3_int_;
  for (std::size_t q = 0; q < n_int; ++q)
    local_sum += static_cast<double>(buf[q]) * static_cast<double>(scale);

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                sub_.topology().cart_comm());
  const int    n_int_global = sub_.n_global()[0] * sub_.n_global()[1] * sub_.n_global()[2];
  const real_t mean_dp      = static_cast<real_t>(global_sum / n_int_global);

  for (int ii = 0; ii < n1_int_; ++ii)
    for (int jj = 0; jj < n2_int_; ++jj)
      for (int kk = 0; kk < n3_int_; ++kk) {
        const real_t val = buf[idx3f(kk, ii, jj, n1_int_, n2_int_)] * scale - mean_dp;
        P.at(ii + h, jj + h, kk + h) = val;
      }

  core::exchange_halo_cpu(P, sub_);
  core::apply_ghost_cpu(P, problem_.P, sub_);
}


// ── project_ ────────────────────────────────────────────────────────────────

void PressureSolverBase::project_(real_t dt,
                                   core::CpuField& U,
                                   core::CpuField& V,
                                   core::CpuField& W,
                                   const core::CpuField& P) {
  real_t*       u = U.data();
  real_t*       v = V.data();
  real_t*       w = W.data();
  const real_t* p = P.data();

  const real_t* dmx1 = grid_.dmx_ptr(Direction::X);
  const real_t* dmx2 = grid_.dmx_ptr(Direction::Y);
  const real_t* dmx3 = grid_.dmx_ptr(Direction::Z);

  const int n2 = n2_tot_, n3 = n3_tot_;
  const int h  = kHaloWidth;

  for (int i = h; i < n1_tot_ - h; ++i)
    for (int j = h; j < n2_tot_ - h; ++j)
      for (int k = h; k < n3_tot_ - h; ++k) {
        u[idx3h(i, j, k, n2, n3)] -= dt * (p[idx3h(i,   j, k, n2, n3)]
                                           - p[idx3h(i-1, j, k, n2, n3)]) / dmx1[i];
        v[idx3h(i, j, k, n2, n3)] -= dt * (p[idx3h(i, j,   k, n2, n3)]
                                           - p[idx3h(i, j-1, k, n2, n3)]) / dmx2[j];
        w[idx3h(i, j, k, n2, n3)] -= dt * (p[idx3h(i, j, k,   n2, n3)]
                                           - p[idx3h(i, j, k-1, n2, n3)]) / dmx3[k];
      }

  core::exchange_halo_cpu(U, sub_);
  core::exchange_halo_cpu(V, sub_);
  core::exchange_halo_cpu(W, sub_);

  core::apply_ghost_cpu(U, problem_.U, sub_);
  core::apply_ghost_cpu(V, problem_.V, sub_);
  core::apply_ghost_cpu(W, problem_.W, sub_);
}

} // namespace mpmstd::equation::pressure
