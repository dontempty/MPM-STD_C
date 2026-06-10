#include "core/halo.hpp"

// CPU implementations of the core communication free functions. Compiled in
// BOTH `make cpu` and `make gpu` (the cpu/ folder is always included); the gpu/
// folder is the one gated to the GPU build. Function selection is by the
// _cpu/_gpu SUFFIX at the call site, not by which build you're in.

namespace mpmstd::core {

void exchange_halo_cpu(CpuField& field, const Subdomain& sub) {
  // Subdomain::exchange_halo wraps the 6 face MPI_Sendrecv calls (periodic wrap
  // handled by the axis communicators). Host buffer here.
  sub.exchange_halo(field.data());
}

void bind_gpu_to_local_rank_cpu(const MpiContext& /*ctx*/) {
  // No GPU on the CPU backend — intentional no-op so backend-parameterized code
  // can call the matching suffix uniformly.
}

} // namespace mpmstd::core
