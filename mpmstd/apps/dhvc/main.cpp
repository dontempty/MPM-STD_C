// apps/dhvc — OB differentially-heated vertical channel (rev.2 §9b Fig 7).
// SKELETON. P6 fills the recipe. Constant properties (NO cross-stress) +
// scalar T + OB Boussinesq buoyancy + Re_δ* diagnostic + tanh stretch.
//   assemble_scalar_system_cpu / solve_scalar_cpu;
//   assemble_momentum_const_visc_cpu + add_buoyancy_force_cpu;
//   … pressure / project; compute_Re_delta_star_cpu (Fig 7).

#include "core/mpi_topology.hpp"

#include <cstdio>

int main(int argc, char** argv) {
  mpmstd::core::MpiContext mpi(&argc, &argv);
  if (mpi.is_root())
    std::printf("[mpmstd] dhvc (skeleton, P0) — links libmpmstd; Fig 7 recipe lands in P6\n");
  return 0;
}
