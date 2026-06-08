// apps/channel — isothermal turbulent channel (rev.2 §7 case A). SKELETON.
// P1 fills the recipe; P0 only proves the app links libmpmstd.
//
// Recipe (no T/mu declared — isothermal ⇒ zero compute & zero memory):
//   while (t < t_end) {
//     dt = compute_cfl_dt_cpu(U,V,W, grid, mpi, dt_cap);
//     assemble_momentum_const_visc_cpu(mom, U,V,W,P, nu, grid,bc,dt);
//     apply_pressure_gradient_cpu     (mom, dpdx, dt);          // physics/forcing
//     solve_momentum_cpu              (mom, U,V,W, dU,dV,dW, mpi);
//     update_velocity_cpu             (U,V,W, dU,dV,dW);
//     apply_mass_flow_correction_cpu  (U, 1.0, mpi, dt);
//     exchange_halo_cpu(U,sub); exchange_halo_cpu(V,sub); exchange_halo_cpu(W,sub);
//     assemble_pressure_system_cpu    (poi, U,V,W, grid,dt);
//     solve_pressure_cpu              (poi, dP, mpi);
//     project_velocity_cpu            (U,V,W,P, dP, grid,dt);
//     exchange_halo_cpu(U,sub);exchange_halo_cpu(V,sub);exchange_halo_cpu(W,sub);exchange_halo_cpu(P,sub);
//     accumulate_statistics_cpu       (stats, U,V,W,P);
//   }

#include "core/mpi_topology.hpp"

#include <cstdio>

int main(int argc, char** argv) {
  mpmstd::core::MpiContext mpi(&argc, &argv);
  if (mpi.is_root())
    std::printf("[mpmstd] channel (skeleton, P0) — links libmpmstd; recipe lands in P1/P2\n");
  return 0;
}
