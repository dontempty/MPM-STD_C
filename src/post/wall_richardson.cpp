#include "post/wall_richardson.hpp"

#include <mpi.h>

#include <cmath>
#include <limits>

// 오케스트레이션 전용 TU — 양쪽 빌드 공용 (열별 |Ri_f| 계산과 가중 보수는 wall_richardson_kernels.{cpp,cu}).

namespace mpmstd::post {

WallRichardsonResult classify_wall_richardson(const field::Field& U, const field::Field& T,
                                              const grid::Grid& grid,
                                              const parallel::Subdomain& sub, const real_t nu,
                                              const real_t alpha, const int side,
                                              const real_t threshold,
                                              const real_t z_plus_reference,
                                              const RichardsonBins& bins,
                                              field::Buffer& weights) {
  MPI_Comm comm = sub.topology().cart_comm();
  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);
  const bool owner = side == 0 ? sub.topology().z().minus_neighbor == MPI_PROC_NULL
                               : sub.topology().z().plus_neighbor == MPI_PROC_NULL;
  int owner_rank = owner ? my_rank : -1;
  MPI_Allreduce(MPI_IN_PLACE, &owner_rank, 1, MPI_INT, MPI_MAX, comm);

  const int n_columns = sub.x().n_local * sub.y().n_local;
  std::vector<real_t> indicator(n_columns, 0);
  std::vector<double> histogram(bins.count + 2, 0.0);
  double low_count = 0;
  if (owner) {
    std::vector<real_t> richardson;
    local_flux_richardson(richardson, U, T, grid, sub, nu, alpha, side, z_plus_reference);
    const double bin_width = (bins.log10_max - bins.log10_min) / bins.count;
    for (int c = 0; c < n_columns; ++c) {
      const double ri = richardson[c];
      if (ri > 0 && ri <= threshold) {
        indicator[c] = 1;
        low_count += 1;
      }
      if (!(ri > 0) || !std::isfinite(ri)) {
        histogram[bins.count + 1] += 1;  // ∞(u_τ = 0)·0·NaN → overflow 칸
        continue;
      }
      const double l = std::log10(ri);
      if (l < bins.log10_min) histogram[0] += 1;
      else if (l >= bins.log10_max) histogram[bins.count + 1] += 1;
      else histogram[1 + static_cast<int>((l - bins.log10_min) / bin_width)] += 1;
    }
  }
  MPI_Bcast(indicator.data(), n_columns, MPI_DOUBLE, owner_rank, comm);
  MPI_Bcast(histogram.data(), bins.count + 2, MPI_DOUBLE, owner_rank, comm);
  MPI_Bcast(&low_count, 1, MPI_DOUBLE, owner_rank, comm);
  if (weights.size() != static_cast<std::size_t>(n_columns)) weights.assign(n_columns);
  weights.upload(indicator);

  WallRichardsonResult out;
  out.area_fraction = static_cast<real_t>(
      low_count / (static_cast<double>(sub.x().n_global) * sub.y().n_global));
  out.histogram = std::move(histogram);
  return out;
}

}  // namespace mpmstd::post
