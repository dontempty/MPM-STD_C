// P-0.5 spike (GPU link/run proof) — GPU build only.
//
//   mpirun -np 2 link_gpu       (on a GPU node; link alone proves the model)
//
// Forces the _gpu symbols to compile + link: allocate a GpuField, bind this
// rank to its node-local GPU (cudaSetDevice(node_rank)), and run a
// device-to-device halo exchange via CUDA-aware MPI. This is the miniature of
// the rev.2 dual-build model — _cpu and _gpu families live in one library; an
// app picks one. Here the binary references the _gpu family.

#include "core/field_gpu.hpp"
#include "core/halo.hpp"
#include "core/mpi_topology.hpp"

#include "parallel/mpi/mpi_context.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <mpi.h>
#include <cstdio>

using namespace mpmstd;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  core::bind_gpu_to_local_rank_gpu(mpi);          // 1 rank = 1 GPU

  const int ws = mpi.world_size();
  parallel::mpi::MpiTopology topo(mpi, {ws, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {8, 4, 4});

  core::GpuField T(sub, "T");
  core::exchange_halo_gpu(T, sub);                // device-to-device CUDA-aware MPI

  if (mpi.is_root())
    std::printf("[PASS] link_gpu: GpuField + bind_gpu_to_local_rank_gpu + exchange_halo_gpu linked & ran (np=%d)\n", ws);
  return 0;
}
