#include "post/statistics.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace mpmstd::post {

// 두 프로파일 함수는 모양이 같지만 템플릿/람다를 쓰지 않는다 — GPU 쌍둥이
// (statistics.cu)와 1:1 대응을 눈으로 맞추기 위해서다.

std::vector<real_t> plane_average(const field::Field& F,
                                  const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int z_offset = sub.z().offset;
  std::vector<real_t> profile(sub.z().n_global, 0.0);
  real_t* prof = profile.data();
  const field::ConstView f = F.view();

  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t value = f(i, j, k);
        const int g = z_offset + (k - h);
        prof[g] += value;
      }
  MPI_Allreduce(MPI_IN_PLACE, prof, static_cast<int>(profile.size()), MPI_DOUBLE,
                MPI_SUM, sub.topology().cart_comm());
  const real_t plane_count =
      static_cast<real_t>(sub.x().n_global) * sub.y().n_global;  // 전역! (16배 버그)
  for (real_t& v : profile) v /= plane_count;
  return profile;
}

std::vector<real_t> plane_average_square(const field::Field& F,
                                         const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int z_offset = sub.z().offset;
  std::vector<real_t> profile(sub.z().n_global, 0.0);
  real_t* prof = profile.data();
  const field::ConstView f = F.view();

  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t value = f(i, j, k) * f(i, j, k);
        const int g = z_offset + (k - h);
        prof[g] += value;
      }
  MPI_Allreduce(MPI_IN_PLACE, prof, static_cast<int>(profile.size()), MPI_DOUBLE,
                MPI_SUM, sub.topology().cart_comm());
  const real_t plane_count =
      static_cast<real_t>(sub.x().n_global) * sub.y().n_global;
  for (real_t& v : profile) v /= plane_count;
  return profile;
}

std::vector<real_t> plane_average_product(const field::Field& F,
                                          const field::Field& G,
                                          const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int z_offset = sub.z().offset;
  std::vector<real_t> profile(sub.z().n_global, 0.0);
  real_t* prof = profile.data();
  const field::ConstView f = F.view();
  const field::ConstView g = G.view();
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) prof[z_offset + (k - h)] += f(i, j, k) * g(i, j, k);
  MPI_Allreduce(MPI_IN_PLACE, prof, static_cast<int>(profile.size()), MPI_DOUBLE,
                MPI_SUM, sub.topology().cart_comm());
  const real_t plane_count =
      static_cast<real_t>(sub.x().n_global) * sub.y().n_global;
  for (real_t& v : profile) v /= plane_count;
  return profile;
}

void span_average_center_moments(const field::Field& U, const field::Field& V,
                                 const field::Field& W, std::vector<real_t>& out,
                                 const parallel::Subdomain& sub) {
  if (sub.topology().y().nprocs != 1)
    throw std::invalid_argument("span_average_center_moments: npy must be 1");
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int nxl = sub.x().n_local, nzl = sub.z().n_local, nyl = sub.y().n_local;
  const std::size_t n = static_cast<std::size_t>(nxl) * nzl;
  out.assign(7 * n, 0.0);
  const field::ConstView u = U.view(), v = V.view(), w = W.view();
  for (int i = h; i < nx - h; ++i)
    for (int k = h; k < nz - h; ++k) {
      real_t s[7] = {0, 0, 0, 0, 0, 0, 0};
      for (int j = h; j < ny - h; ++j) {
        const real_t uc = real_t(0.5) * (u(i, j, k) + u(i + 1, j, k));
        const real_t vc = real_t(0.5) * (v(i, j, k) + v(i, j + 1, k));
        const real_t wc = real_t(0.5) * (w(i, j, k) + w(i, j, k + 1));
        s[0] += uc; s[1] += vc; s[2] += wc;
        s[3] += uc * uc; s[4] += vc * vc; s[5] += wc * wc; s[6] += uc * wc;
      }
      const std::size_t q = static_cast<std::size_t>(i - h) * nzl + (k - h);
      for (int m = 0; m < 7; ++m) out[m * n + q] = s[m] / nyl;
    }
}

real_t bulk_velocity(const field::Field& U, const grid::Grid& grid,
                     const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const field::ConstView u = U.view();
  const real_t* dx = grid.x.cell_width.data();
  const real_t* dy = grid.y.cell_width.data();
  const real_t* dz = grid.z.cell_width.data();

  real_t sum_flux = 0;
  real_t sum_volume = 0;
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t volume = dx[i] * dy[j] * dz[k];
        sum_flux += real_t(0.5) * (u(i, j, k) + u(i + 1, j, k)) * volume;
        sum_volume += volume;
      }
  real_t local[2] = {sum_flux, sum_volume};
  real_t total[2] = {0, 0};
  MPI_Allreduce(local, total, 2, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  return (total[1] > 0) ? total[0] / total[1] : real_t(0);
}

real_t max_divergence(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub) {
  const int h = halo_width;
  const field::ConstView u = U.view();
  const field::ConstView v = V.view();
  const field::ConstView w = W.view();
  real_t worst = 0;
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        const real_t divergence =
            std::abs((u(i + 1, j, k) - u(i, j, k)) / grid.x.cell_width[i]
                     + (v(i, j + 1, k) - v(i, j, k)) / grid.y.cell_width[j]
                     + (w(i, j, k + 1) - w(i, j, k)) / grid.z.cell_width[k]);
        worst = std::max(worst, divergence);
      }
  real_t global = 0;
  MPI_Allreduce(&worst, &global, 1, MPI_DOUBLE, MPI_MAX, sub.topology().cart_comm());
  return global;
}


// ── 열 가중 평면 평균 (2026-10-07; GPU 쌍둥이 statistics.cu 와 1:1) ─────────────────────────
namespace {

std::vector<real_t> weighted_profile_impl(const field::Field& F, const field::Field* G,
                                          const field::Buffer& weights,
                                          const parallel::Subdomain& sub, const bool squared) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int i2 = sub.y().n_local;
  const int z_offset = sub.z().offset;
  if (weights.size() != static_cast<std::size_t>(sub.x().n_local) * i2)
    throw std::invalid_argument("plane_weighted_average: weights size != nx_local*ny_local");
  std::vector<real_t> profile(sub.z().n_global, 0.0);
  real_t* prof = profile.data();
  const real_t* w = weights.data();
  const field::ConstView f = F.view();
  const field::ConstView g = G ? G->view() : F.view();
  real_t weight_sum = 0;
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j) {
      const real_t wij = w[static_cast<std::size_t>(i - h) * i2 + (j - h)];
      weight_sum += wij;
      if (wij == 0) continue;
      for (int k = h; k < nz - h; ++k) {
        const real_t value = squared ? f(i, j, k) * f(i, j, k)
                                     : (G ? f(i, j, k) * g(i, j, k) : f(i, j, k));
        prof[z_offset + (k - h)] += wij * value;
      }
    }
  MPI_Allreduce(MPI_IN_PLACE, prof, static_cast<int>(profile.size()), MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
  // Σw 는 평면(x,y) 분할 랭크들의 합 — z 분할은 같은 평면을 공유하므로 z 축 랭크 수로 나눈다
  MPI_Allreduce(MPI_IN_PLACE, &weight_sum, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  weight_sum /= static_cast<real_t>(sub.topology().z().nprocs);
  if (weight_sum > 0)
    for (real_t& v : profile) v /= weight_sum;
  else
    for (real_t& v : profile) v = 0;
  return profile;
}

}  // namespace

std::vector<real_t> plane_weighted_average(const field::Field& F, const field::Buffer& weights,
                                           const parallel::Subdomain& sub) {
  return weighted_profile_impl(F, nullptr, weights, sub, /*squared=*/false);
}

std::vector<real_t> plane_weighted_average_square(const field::Field& F,
                                                  const field::Buffer& weights,
                                                  const parallel::Subdomain& sub) {
  return weighted_profile_impl(F, nullptr, weights, sub, /*squared=*/true);
}

std::vector<real_t> plane_weighted_average_product(const field::Field& F, const field::Field& G,
                                                   const field::Buffer& weights,
                                                   const parallel::Subdomain& sub) {
  return weighted_profile_impl(F, &G, weights, sub, /*squared=*/false);
}

}  // namespace mpmstd::post
