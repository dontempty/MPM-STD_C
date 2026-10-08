#include "post/spatial_statistics.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "post/statistics.hpp"

namespace mpmstd::post {

SpatialStatistics::SpatialStatistics(const grid::Grid& grid, const parallel::Subdomain& sub,
                                     const std::vector<real_t>& z_center_global,
                                     const std::vector<real_t>& z_face_global,
                                     const real_t u_infinity, const real_t nu,
                                     const real_t relaxation_time)
    : grid_(grid), sub_(sub), zc_(z_center_global), zf_(z_face_global), u_inf_(u_infinity),
      nu_(nu), tau_(relaxation_time), nxl_(sub.x().n_local), nzl_(sub.z().n_local),
      nz_global_(sub.z().n_global) {
  if (sub.topology().x().nprocs != 1 || sub.topology().y().nprocs != 1)
    throw std::invalid_argument("SpatialStatistics: z-slab decomposition only");
  relaxed_u_.assign(static_cast<std::size_t>(nxl_) * nzl_, u_inf_);
  sums_.assign(7 * static_cast<std::size_t>(nxl_) * nzl_, 0.0);
}

void SpatialStatistics::sample(const field::Field& U, const field::Field& V,
                               const field::Field& W, const real_t dt, const bool accumulate) {
  std::vector<real_t> m;
  span_average_center_moments(U, V, W, m, sub_);
  const std::size_t n = static_cast<std::size_t>(nxl_) * nzl_;
  if (!relaxed_ready_) {
    for (std::size_t q = 0; q < n; ++q) relaxed_u_[q] = m[q];
    relaxed_ready_ = true;
  } else {
    const real_t a = std::min(real_t(1), dt / tau_);
    for (std::size_t q = 0; q < n; ++q) relaxed_u_[q] = (1 - a) * relaxed_u_[q] + a * m[q];
  }
  if (accumulate) {
    for (std::size_t q = 0; q < 7 * n; ++q) sums_[q] += dt * m[q];
    time_sum_ += dt;
  }
}

std::vector<real_t> SpatialStatistics::gather_z(const std::vector<real_t>& local) const {
  MPI_Comm comm = sub_.topology().z().comm;
  int nprocs = 0;
  MPI_Comm_size(comm, &nprocs);
  std::vector<int> nz_each(nprocs), off_each(nprocs), counts(nprocs), displs(nprocs);
  MPI_Allgather(&nzl_, 1, MPI_INT, nz_each.data(), 1, MPI_INT, comm);
  const int my_off = sub_.z().offset;
  MPI_Allgather(&my_off, 1, MPI_INT, off_each.data(), 1, MPI_INT, comm);
  int total = 0;
  for (int r = 0; r < nprocs; ++r) { counts[r] = nxl_ * nz_each[r]; displs[r] = total; total += counts[r]; }
  std::vector<real_t> all(total);
  MPI_Allgatherv(local.data(), nxl_ * nzl_, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                 MPI_DOUBLE, comm);
  std::vector<real_t> out(static_cast<std::size_t>(nxl_) * nz_global_);
  for (int r = 0; r < nprocs; ++r)
    for (int i = 0; i < nxl_; ++i)
      for (int k = 0; k < nz_each[r]; ++k)
        out[static_cast<std::size_t>(i) * nz_global_ + off_each[r] + k] =
            all[displs[r] + static_cast<std::size_t>(i) * nz_each[r] + k];
  return out;
}

BoundaryLayerIntegrals SpatialStatistics::integrals_of(const std::vector<real_t>& mu) const {
  const int h = halo_width;
  BoundaryLayerIntegrals r;
  const int nx = nxl_;
  r.x.resize(nx); r.cf.resize(nx); r.delta99.resize(nx); r.dstar.resize(nx); r.theta.resize(nx);
  r.shape.resize(nx); r.re_theta.resize(nx); r.re_tau.resize(nx); r.u_tau.resize(nx);
  r.u_edge.resize(nx);
  const real_t z1 = zc_[0], z2 = zc_[1];
  const int n_edge = std::max(1, nz_global_ / 20);
  for (int i = 0; i < nx; ++i) {
    const real_t* u = &mu[static_cast<std::size_t>(i) * nz_global_];
    r.x[i] = grid_.x.center[i + h];
    real_t u_edge = 0;  // 국소 자유류 속도 (상단 5% 셀 평균; 0 이면 U∞)
    for (int k = nz_global_ - n_edge; k < nz_global_; ++k) u_edge += u[k];
    u_edge /= static_cast<real_t>(n_edge);
    if (!(u_edge > 0)) u_edge = u_inf_;
    r.u_edge[i] = u_edge;
    const real_t dudz = u[0] * z2 / (z1 * (z2 - z1)) - u[1] * z1 / (z2 * (z2 - z1));
    const real_t u_tau = std::sqrt(std::max(real_t(0), nu_ * dudz));
    r.u_tau[i] = u_tau;
    r.cf[i] = 2 * nu_ * dudz / (u_edge * u_edge);
    real_t dstar = 0, theta = 0;
    for (int k = 0; k < nz_global_; ++k) {
      const real_t f = std::min(real_t(1), std::max(real_t(0), u[k] / u_edge));
      const real_t dz = zf_[k + 1] - zf_[k];
      dstar += (1 - f) * dz;
      theta += f * (1 - f) * dz;
    }
    real_t d99 = zc_.back();
    for (int k = 0; k < nz_global_; ++k)
      if (u[k] >= real_t(0.99) * u_edge) {
        if (k == 0) { d99 = zc_[0]; break; }
        const real_t w = (real_t(0.99) * u_edge - u[k - 1]) / (u[k] - u[k - 1]);
        d99 = zc_[k - 1] + w * (zc_[k] - zc_[k - 1]);
        break;
      }
    r.dstar[i] = dstar; r.theta[i] = theta; r.delta99[i] = d99;
    r.shape[i] = theta > 0 ? dstar / theta : 0;
    r.re_theta[i] = u_edge * theta / nu_;
    r.re_tau[i] = u_tau * d99 / nu_;
  }
  return r;
}

BoundaryLayerIntegrals SpatialStatistics::integrals(const bool from_relaxed) const {
  std::vector<real_t> local(static_cast<std::size_t>(nxl_) * nzl_);
  if (from_relaxed) local = relaxed_u_;
  else {
    const double inv = time_sum_ > 0 ? 1.0 / time_sum_ : 0.0;
    for (std::size_t q = 0; q < local.size(); ++q) local[q] = sums_[q] * inv;
  }
  return integrals_of(gather_z(local));
}

std::vector<real_t> SpatialStatistics::free_stream_w(const BoundaryLayerIntegrals& in) const {
  // δ*(x) ≈ a + b x + c x² (0.1 ≤ x/Lx ≤ 0.9, 정규방정식) → w_top = U∞ (b + 2 c x)
  const int nx = nxl_;
  const real_t lx = grid_.x.face[halo_width + nx];
  double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
  for (int i = 0; i < nx; ++i) {
    const double x = in.x[i];
    if (x < 0.1 * lx || x > 0.9 * lx) continue;
    const double p[5] = {1, x, x * x, x * x * x, x * x * x * x};
    for (int m = 0; m < 5; ++m) S[m] += p[m];
    T[0] += in.dstar[i]; T[1] += in.dstar[i] * x; T[2] += in.dstar[i] * x * x;
  }
  // 3×3 풀이 (Cramer)
  const double A[3][3] = {{S[0], S[1], S[2]}, {S[1], S[2], S[3]}, {S[2], S[3], S[4]}};
  const auto det3 = [](const double m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  };
  const double D = det3(A);
  double coef[3] = {0, 0, 0};
  if (std::abs(D) > 1e-300)
    for (int c = 0; c < 3; ++c) {
      double M[3][3];
      for (int r = 0; r < 3; ++r) for (int q = 0; q < 3; ++q) M[r][q] = (q == c) ? T[r] : A[r][q];
      coef[c] = det3(M) / D;
    }
  std::vector<real_t> w(sub_.x().n_total, 0.0);
  for (int i = 0; i < sub_.x().n_total; ++i) {
    const double x = grid_.x.center[i];
    w[i] = static_cast<real_t>(std::max(0.0, u_inf_ * (coef[1] + 2 * coef[2] * x)));
  }
  return w;
}

void SpatialStatistics::write(const std::string& path, const long step, const double time,
                              const std::vector<real_t>& station_re_theta) const {
  const std::size_t n = static_cast<std::size_t>(nxl_) * nzl_;
  const double inv = time_sum_ > 0 ? 1.0 / time_sum_ : 0.0;
  std::vector<std::vector<real_t>> g(7);
  for (int m = 0; m < 7; ++m) {
    std::vector<real_t> local(n);
    for (std::size_t q = 0; q < n; ++q) local[q] = sums_[m * n + q] * inv;
    g[m] = gather_z(local);
  }
  const BoundaryLayerIntegrals in = integrals_of(g[0]);
  int rank = 0;
  MPI_Comm_rank(sub_.topology().cart_comm(), &rank);
  if (rank != 0) return;
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  std::fprintf(f, "# boundary layer statistics  step=%ld t=%.6f nu=%.10e u_inf=%g accumulated_time=%.6f nx=%d nz=%d\n",
               step, time, nu_, u_inf_, time_sum_, nxl_, nz_global_);
  std::fprintf(f, "# section 1: x  Re_x_rel(x/nu)  Cf  delta99  dstar  theta  H  Re_theta  Re_tau  u_tau  u_edge"
                  "   (Cf, dstar, theta, delta99, Re_theta normalised by the local edge velocity u_edge)\n");
  for (int i = 0; i < nxl_; ++i)
    std::fprintf(f, "%12.6f %14.6e %12.6e %12.6f %12.6f %12.6f %10.5f %12.4f %12.4f %12.6e %12.6f\n",
                 in.x[i], u_inf_ * in.x[i] / nu_, in.cf[i], in.delta99[i], in.dstar[i], in.theta[i],
                 in.shape[i], in.re_theta[i], in.re_tau[i], in.u_tau[i], in.u_edge[i]);
  for (const real_t target : station_re_theta) {
    int best = -1;
    for (int i = 0; i < nxl_; ++i)
      if (best < 0 || std::abs(in.re_theta[i] - target) < std::abs(in.re_theta[best] - target)) best = i;
    if (best < 0) continue;
    const real_t ut = in.u_tau[best];
    if (ut <= 0) continue;
    std::fprintf(f, "\n\n# section 2: station target Re_theta=%g -> i=%d x=%.4f Re_theta=%.2f Re_tau=%.2f Cf=%.6e H=%.4f delta99=%.4f u_tau=%.6e u_edge=%.6f\n",
                 target, best, in.x[best], in.re_theta[best], in.re_tau[best], in.cf[best], in.shape[best],
                 in.delta99[best], ut, in.u_edge[best]);
    std::fprintf(f, "# z  z+  U+  urms+  vrms+  wrms+  uw+\n");
    const std::size_t b = static_cast<std::size_t>(best) * nz_global_;
    for (int k = 0; k < nz_global_; ++k) {
      const real_t um = g[0][b + k], vm = g[1][b + k], wm = g[2][b + k];
      const real_t uu = std::max(real_t(0), g[3][b + k] - um * um);
      const real_t vv = std::max(real_t(0), g[4][b + k] - vm * vm);
      const real_t ww = std::max(real_t(0), g[5][b + k] - wm * wm);
      const real_t uw = g[6][b + k] - um * wm;
      std::fprintf(f, "%12.6f %12.4f %12.6f %12.6f %12.6f %12.6f %12.6f\n", zc_[k], zc_[k] * ut / nu_,
                   um / ut, std::sqrt(uu) / ut, std::sqrt(vv) / ut, std::sqrt(ww) / ut, uw / (ut * ut));
    }
  }
  std::fclose(f);
}

}  // namespace mpmstd::post
