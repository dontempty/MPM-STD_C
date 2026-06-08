// apps/rbc — NOB Rayleigh-Bénard, glycerol (rev.2 §7 case B, §9b Fig 9). SKELETON.
// P7 fills the NOB recipe; P0 only proves the app links libmpmstd.
//
// Recipe (T + variable properties + buoyancy declared ONLY here):
//   update_properties_cpu        (mu,irho,kappa,irhocp, T, model);   // NOB μ(T)…
//   assemble_scalar_system_cpu   (Tsys, T,U,V,W, kappa, grid,bc,dt); exchange_halo_cpu(T,sub);
//   solve_scalar_cpu             (Tsys, T, mpi); exchange_halo_cpu(T,sub);
//   assemble_momentum_var_visc_cpu(mom, U,V,W,P, mu, grid,bc,dt);    // cross-stress
//   add_buoyancy_force_cpu       (mom, T, buoy, dt);
//   solve_momentum_cpu / update_velocity_cpu / pressure / project / stats …

#include "core/mpi_topology.hpp"

#include <cstdio>

int main(int argc, char** argv) {
  mpmstd::core::MpiContext mpi(&argc, &argv);
  if (mpi.is_root())
    std::printf("[mpmstd] rbc (skeleton, P0) — links libmpmstd; NOB recipe lands in P7\n");
  return 0;
}
