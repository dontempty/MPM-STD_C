#include "post/scalar_spatial_statistics.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "post/statistics.hpp"

namespace mpmstd::post {

ScalarSpatialStatistics::ScalarSpatialStatistics(
    const grid::Grid& grid, const parallel::Subdomain& sub,
    const std::vector<real_t>& z_center_global, const std::vector<real_t>& z_face_global,
    const real_t nu, const real_t prandtl, const ScalarWall wall, const real_t wall_value,
    const real_t wall_gradient, const real_t theta_infinity)
    : grid_(grid), sub_(sub), zc_(z_center_global), zf_(z_face_global), nu_(nu),
      prandtl_(prandtl), alpha_(nu / prandtl), wall_(wall), wall_value_(wall_value),
      wall_gradient_(wall_gradient), theta_inf_(theta_infinity), nxl_(sub.x().n_local),
      nzl_(sub.z().n_local), nz_global_(sub.z().n_global) {
  if (sub.topology().x().nprocs != 1 || sub.topology().y().nprocs != 1)
    throw std::invalid_argument("ScalarSpatialStatistics: z-slab decomposition only");
  sums_.assign(6 * static_cast<std::size_t>(nxl_) * nzl_, 0.0);
}

void ScalarSpatialStatistics::sample(const field::Field& U, const field::Field& W,
                                     const field::Field& T, const real_t dt,
                                     const bool accumulate) {
  if (!accumulate) return;
  std::vector<real_t> m;
  span_average_scalar_moments(U, W, T, m, sub_);
  for (std::size_t q = 0; q < sums_.size(); ++q) sums_[q] += dt * m[q];
  time_sum_ += dt;
}

std::vector<real_t> ScalarSpatialStatistics::gather_z(const std::vector<real_t>& local) const {
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

void ScalarSpatialStatistics::wall_state(const real_t* theta, real_t& theta_wall,
                                         real_t& gradient_wall) const {
  const real_t z1 = zc_[0], z2 = zc_[1];
  if (wall_ == ScalarWall::Isoscalar) {
    theta_wall = wall_value_;
    // 2점 Lagrange 편측 미분 (θ(0) = θ_w 포함 3점 — 속도의 u_tau 와 같은 자)
    const real_t t1 = theta[0] - theta_wall, t2 = theta[1] - theta_wall;
    gradient_wall = t1 * z2 / (z1 * (z2 - z1)) - t2 * z1 / (z2 * (z2 - z1));
  } else {
    // 구배 처방: θ(z) = θ_w + g z + c z² 를 두 셀에 맞춰 θ_w 외삽
    gradient_wall = wall_gradient_;
    const real_t g = wall_gradient_;
    const real_t c = ((theta[1] - theta[0]) - g * (z2 - z1)) / (z2 * z2 - z1 * z1);
    theta_wall = theta[0] - g * z1 - c * z1 * z1;
  }
}

ScalarIntegrals ScalarSpatialStatistics::integrals_of(const std::vector<real_t>& mt,
                                                      const BoundaryLayerIntegrals& v) const {
  const int h = halo_width;
  ScalarIntegrals r;
  const int nx = nxl_;
  r.x.resize(nx); r.stanton.resize(nx); r.theta_wall.resize(nx); r.gradient_wall.resize(nx);
  r.delta99_theta.resize(nx); r.theta_tau.resize(nx); r.reynolds_analogy.resize(nx);
  for (int i = 0; i < nx; ++i) {
    const real_t* t = &mt[static_cast<std::size_t>(i) * nz_global_];
    r.x[i] = grid_.x.center[i + h];
    real_t tw = 0, gw = 0;
    wall_state(t, tw, gw);
    r.theta_wall[i] = tw;
    r.gradient_wall[i] = gw;
    const real_t difference = theta_inf_ - tw;
    const real_t u_edge = v.u_edge[i] > 0 ? v.u_edge[i] : real_t(1);
    r.stanton[i] = std::abs(difference) > 0 ? alpha_ * gw / (u_edge * difference) : 0;
    r.theta_tau[i] = v.u_tau[i] > 0 ? alpha_ * gw / v.u_tau[i] : 0;
    r.reynolds_analogy[i] = v.cf[i] > 0 ? 2 * r.stanton[i] / v.cf[i] : 0;
    real_t d99 = zc_.back();
    for (int k = 0; k < nz_global_; ++k) {
      const real_t f = (t[k] - tw) / (std::abs(difference) > 0 ? difference : real_t(1));
      if (f >= real_t(0.99)) {
        if (k == 0) { d99 = zc_[0]; break; }
        const real_t fp = (t[k - 1] - tw) / difference;
        const real_t w = (real_t(0.99) - fp) / (f - fp);
        d99 = zc_[k - 1] + w * (zc_[k] - zc_[k - 1]);
        break;
      }
    }
    r.delta99_theta[i] = d99;
  }
  return r;
}

ScalarIntegrals ScalarSpatialStatistics::integrals(const BoundaryLayerIntegrals& velocity) const {
  const std::size_t n = static_cast<std::size_t>(nxl_) * nzl_;
  std::vector<real_t> local(n);
  const double inv = time_sum_ > 0 ? 1.0 / time_sum_ : 0.0;
  for (std::size_t q = 0; q < n; ++q) local[q] = sums_[q] * inv;
  return integrals_of(gather_z(local), velocity);
}

void ScalarSpatialStatistics::write(const std::string& path, const long step, const double time,
                                    const std::vector<real_t>& station_re_theta,
                                    const BoundaryLayerIntegrals& velocity) const {
  const std::size_t n = static_cast<std::size_t>(nxl_) * nzl_;
  const double inv = time_sum_ > 0 ? 1.0 / time_sum_ : 0.0;
  std::vector<std::vector<real_t>> g(6);
  for (int m = 0; m < 6; ++m) {
    std::vector<real_t> local(n);
    for (std::size_t q = 0; q < n; ++q) local[q] = sums_[m * n + q] * inv;
    g[m] = gather_z(local);
  }
  const ScalarIntegrals in = integrals_of(g[0], velocity);
  int rank = 0;
  MPI_Comm_rank(sub_.topology().cart_comm(), &rank);
  if (rank != 0) return;
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  std::fprintf(f, "# boundary layer scalar statistics  step=%ld t=%.6f nu=%.10e Pr=%g alpha=%.10e wall=%s "
                  "wall_value=%g wall_gradient=%g theta_inf=%g accumulated_time=%.6f nx=%d nz=%d\n",
               step, time, nu_, prandtl_, alpha_, wall_ == ScalarWall::Isoscalar ? "isoscalar" : "isoflux",
               wall_value_, wall_gradient_, theta_inf_, time_sum_, nxl_, nz_global_);
  std::fprintf(f, "# section 1: x  Re_theta  St  theta_wall  dtheta_dz_wall  delta99_theta  theta_tau  2St_over_Cf"
                  "   (St = alpha*dtheta_dz_wall/(u_edge*(theta_inf-theta_wall)); theta_tau = alpha*dtheta_dz_wall/u_tau)\n");
  for (int i = 0; i < nxl_; ++i)
    std::fprintf(f, "%12.6f %12.4f %12.6e %12.6f %12.6e %12.6f %12.6e %12.6f\n", in.x[i],
                 velocity.re_theta[i], in.stanton[i], in.theta_wall[i], in.gradient_wall[i],
                 in.delta99_theta[i], in.theta_tau[i], in.reynolds_analogy[i]);
  for (const real_t target : station_re_theta) {
    int best = -1;
    for (int i = 0; i < nxl_; ++i)
      if (best < 0 || std::abs(velocity.re_theta[i] - target) < std::abs(velocity.re_theta[best] - target))
        best = i;
    if (best < 0) continue;
    const real_t ut = velocity.u_tau[best], tt = in.theta_tau[best];
    if (ut <= 0 || tt == 0) continue;
    std::fprintf(f, "\n\n# section 2: station target Re_theta=%g -> i=%d x=%.4f Re_theta=%.2f Re_tau=%.2f u_tau=%.6e "
                    "St=%.6e theta_wall=%.6f theta_tau=%.6e delta99_theta=%.4f\n",
                 target, best, in.x[best], velocity.re_theta[best], velocity.re_tau[best], ut,
                 in.stanton[best], in.theta_wall[best], tt, in.delta99_theta[best]);
    std::fprintf(f, "# z  z+  theta+  theta_rms+  ut+  wt+  theta_mean  U_c_mean   (theta+ = (theta-theta_w)/theta_tau; "
                    "ut+ = <u'theta'>/(u_tau theta_tau), wt+ = <w'theta'>/(u_tau theta_tau), w = wall-normal)\n");
    const std::size_t b = static_cast<std::size_t>(best) * nz_global_;
    const real_t tw = in.theta_wall[best];
    for (int k = 0; k < nz_global_; ++k) {
      const real_t tm = g[0][b + k], um = g[2][b + k], wm = g[3][b + k];
      const real_t var = std::max(real_t(0), g[1][b + k] - tm * tm);
      const real_t ut_flux = g[4][b + k] - um * tm;
      const real_t wt_flux = g[5][b + k] - wm * tm;
      std::fprintf(f, "%12.6f %12.4f %12.6f %12.6f %12.6e %12.6e %12.6f %12.6f\n", zc_[k],
                   zc_[k] * ut / nu_, (tm - tw) / tt, std::sqrt(var) / std::abs(tt),
                   ut_flux / (ut * tt), wt_flux / (ut * tt), tm, um);
    }
  }
  std::fclose(f);
}

}  // namespace mpmstd::post
