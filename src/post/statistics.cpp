#include "post/statistics.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>


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

}  // namespace mpmstd::post
