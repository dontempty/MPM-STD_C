#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"   // Subdomain

namespace mpmstd::core {

// =============================================================================
// Ghost-cell exchange — explicit free function shown AROUND solve (rev.2 C3).
// -----------------------------------------------------------------------------
// API-freeze decision (resolves plan §5 "API mismatch #1"): the realized
// signature takes a `Subdomain`, not an `MpiTopology`. The per-axis MPI derived
// datatypes + neighbour ranks live in Subdomain (Subdomain::exchange_halo), so
// passing it is both natural and sufficient; MpiTopology is reachable via
// Subdomain::topology() if ever needed.
//
//   _cpu : exchanges the host buffer.
//   _gpu : exchanges the DEVICE buffer directly via CUDA-aware MPI (rev.2 C2);
//          no host staging. Defined in core/gpu/halo_gpu.cu.
//
// Faces only (no edges/corners) — matches the underlying Subdomain routine;
// extend there if a cross-derivative stencil ever needs corners.
// =============================================================================
void exchange_halo_cpu(CpuField& f, const Subdomain& sub);
void exchange_halo_gpu(GpuField& f, const Subdomain& sub);

} // namespace mpmstd::core
