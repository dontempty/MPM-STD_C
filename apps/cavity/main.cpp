// apps/cavity — lid-driven cavity (rev.2 §9c, P3b: arbitrary-BC + Poisson
// generalization vs Ghia et al.). SKELETON. Same isothermal recipe as channel;
// differs ONLY by the input BC (all walls + moving lid, no periodic axis →
// all-Neumann pressure ⇒ DCT+TDMA + null-space pin).

#include "core/mpi_topology.hpp"

#include <cstdio>

int main(int argc, char** argv) {
  mpmstd::core::MpiContext mpi(&argc, &argv);
  if (mpi.is_root())
    std::printf("[mpmstd] cavity (skeleton, P0) — links libmpmstd; validated in P3b\n");
  return 0;
}
