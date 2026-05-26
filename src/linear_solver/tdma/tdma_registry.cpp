#include "linear_solver/tdma/tdma_registry.hpp"

#include "linear_solver/tdma/pascal_tdma_cpu_backend.hpp"
#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"

namespace mpmstd::linear_solver::tdma {

std::unique_ptr<TdmaRegistry>
TdmaRegistry::make_default(const parallel::mpi::MpiTopology& topo) {
  auto reg = std::make_unique<TdmaRegistry>();
  for (int a = 0; a < 3; ++a) {
    const auto& axis = topo.axis(static_cast<Direction>(a));
#ifdef MPMSTD_BACKEND_CUDA
    reg->backends_[a] = std::make_unique<PaScaLTDMACudaBackend>(axis);
#else
    reg->backends_[a] = std::make_unique<PaScaLTDMACpuBackend>(axis);
#endif
  }
  return reg;
}

} // namespace mpmstd::linear_solver::tdma
