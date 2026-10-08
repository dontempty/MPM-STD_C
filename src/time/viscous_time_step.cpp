#include "time/viscous_time_step.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>

#include "post/statistics.hpp"

// 양쪽 빌드 공용 호스트 TU — 평면 최대는 post::plane_maximum(CPU/GPU 쌍둥이)이 한다.

namespace mpmstd::time_integration {

ViscousCouplingLimit compute_viscous_coupling_dt(const field::Field& viscosity,
                                                 const grid::Grid& grid,
                                                 const parallel::Subdomain& sub,
                                                 const real_t Cmu, const real_t coupling_number,
                                                 const real_t dt_reference) {
  const int h = halo_width;
  const std::vector<real_t> mu_max = post::plane_maximum(viscosity, sub, /*absolute=*/false);
  // Δ_h = 내부 최소 수평 간격 (x·y 균일 전제지만 일반적으로 최소를 취한다)
  real_t dh = std::numeric_limits<real_t>::infinity();
  for (int i = h; i < sub.x().n_total - h; ++i) dh = std::min(dh, grid.x.cell_width[i]);
  for (int j = h; j < sub.y().n_total - h; ++j) dh = std::min(dh, grid.y.cell_width[j]);
  ViscousCouplingLimit out;
  real_t worst_rate = 0;  // max_k C_μ μ̂_max,k / (Δ_h Δz_k)  — 로컬 z 구간만 보고 뒤에 Allreduce
  for (int k = h; k < sub.z().n_total - h; ++k) {
    const int kg = sub.z().global(k);
    const real_t rate = Cmu * mu_max[kg] / (dh * grid.z.cell_width[k]);
    if (rate > worst_rate) {
      worst_rate = rate;
      out.k_global = kg;
      out.viscosity_max = mu_max[kg];
    }
  }
  // 랭크마다 자기 z 구간의 최악을 봤다 → 전역 최악으로 (k·μ̂ 는 최악 랭크의 것을 Bcast)
  struct { double rate; int rank; } local{static_cast<double>(worst_rate), 0}, global{};
  MPI_Comm comm = sub.topology().cart_comm();
  MPI_Comm_rank(comm, &local.rank);
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  MPI_Bcast(&out.k_global, 1, MPI_INT, global.rank, comm);
  MPI_Bcast(&out.viscosity_max, 1, MPI_DOUBLE, global.rank, comm);
  worst_rate = static_cast<real_t>(global.rate);
  out.dt = worst_rate > 0 ? coupling_number / worst_rate : std::numeric_limits<real_t>::infinity();
  out.mixed_number = dt_reference > 0 ? worst_rate * dt_reference : 0;
  return out;
}

}  // namespace mpmstd::time_integration
