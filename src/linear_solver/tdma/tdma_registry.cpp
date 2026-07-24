#include "linear_solver/tdma/tdma_registry.hpp"

namespace mpmstd::linear_solver::tdma {

std::unique_ptr<TdmaRegistry>
TdmaRegistry::make_default(const parallel::mpi::MpiTopology& topo) {
  auto reg = std::make_unique<TdmaRegistry>();
  for (int a = 0; a < 3; ++a) {
    const auto& axis = topo.axis(static_cast<Direction>(a));
    reg->backends_[a] = std::make_unique<TdmaBackend>(axis);
  }
  return reg;
}

} // namespace mpmstd::linear_solver::tdma
