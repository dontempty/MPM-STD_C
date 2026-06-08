// P-0.5 spike (REQUIRED) — CpuField + exchange_halo_cpu, 2-process halo.
//
//   mpirun -np 2 halo_poc      (works for any np >= 2)
//
// Each interior cell is initialised with a globally-unique value encode(gi,gj,gk).
// After exchange_halo_cpu, every FACE halo cell must equal encode evaluated at
// the neighbouring rank's interior position (periodic wrap via the axis comms).
// This validates the NEW core API (CpuField + free-function halo + Subdomain)
// against the proven behaviour of the old ScalarField/Subdomain path.

#include "core/field_cpu.hpp"
#include "core/halo.hpp"
#include "core/mpi_topology.hpp"

#include "parallel/mpi/mpi_context.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>

using namespace mpmstd;

static inline double encode(int gi, int gj, int gk) {
  return gi * 1000.0 + gj * 1.0 + gk * 0.001;
}

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  const int ws = mpi.world_size();
  if (ws < 2) {
    if (mpi.is_root()) std::fprintf(stderr, "halo_poc needs >=2 ranks (got %d)\n", ws);
    return 2;
  }

  // Decompose along x only (y,z single-rank → periodic self-wrap).
  parallel::mpi::MpiTopology topo(mpi, {ws, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {8, 4, 4});

  core::CpuField T(sub, "T");

  const auto nt  = sub.n_total();
  const auto ni  = sub.n_interior();
  const auto off = sub.global_offset();
  const std::array<int, 3> ng = sub.n_global();

  // fill interior with global-encoded value
  for (int i = 0; i < ni[0]; ++i)
    for (int j = 0; j < ni[1]; ++j)
      for (int k = 0; k < ni[2]; ++k)
        T.at(i + kHaloWidth, j + kHaloWidth, k + kHaloWidth) =
            encode(off[0] + i, off[1] + j, off[2] + k);

  core::exchange_halo_cpu(T, sub);

  // verify the 6 FACE-halo planes (kHaloWidth == 1: minus at idx 0, plus at nt-1)
  int fails = 0;
  for (int i = 0; i < nt[0]; ++i)
    for (int j = 0; j < nt[1]; ++j)
      for (int k = 0; k < nt[2]; ++k) {
        const int loc[3] = {i, j, k};
        int halo_axis = -1, halo_count = 0;
        for (int a = 0; a < 3; ++a)
          if (loc[a] < kHaloWidth || loc[a] >= nt[a] - kHaloWidth) { ++halo_count; halo_axis = a; }
        if (halo_count != 1) continue;   // skip interior + edge/corner (faces only)

        int g[3];
        for (int a = 0; a < 3; ++a) {
          if (a == halo_axis)
            g[a] = (loc[a] < kHaloWidth) ? (off[a] - 1 + ng[a]) % ng[a]
                                         : (off[a] + ni[a]) % ng[a];
          else
            g[a] = off[a] + (loc[a] - kHaloWidth);
        }
        if (std::abs(T.at(i, j, k) - encode(g[0], g[1], g[2])) > 1e-12) ++fails;
      }

  int global_fails = 0;
  MPI_Allreduce(&fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (mpi.is_root()) {
    if (global_fails == 0)
      std::printf("[PASS] halo_poc: CpuField + exchange_halo_cpu (np=%d, periodic) — all face halos correct\n", ws);
    else
      std::printf("[FAIL] halo_poc: %d halo mismatches across ranks\n", global_fails);
  }
  return global_fails == 0 ? 0 : 1;
}
