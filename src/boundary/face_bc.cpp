#include "boundary/face_bc.hpp"

namespace mpmstd::boundary {

FaceBc FaceBc::periodic() {
  return FaceBc{BcKind::Periodic, real_t{0}, GhostPolicy::ZeroGhost};
}
FaceBc FaceBc::dirichlet(real_t v) {
  return FaceBc{BcKind::Dirichlet, v, GhostPolicy::ZeroGhost};
}
FaceBc FaceBc::dirichlet_antisymm(real_t v) {
  return FaceBc{BcKind::Dirichlet, v, GhostPolicy::Antisymmetric};
}
FaceBc FaceBc::neumann(real_t v) {
  return FaceBc{BcKind::Neumann, v, GhostPolicy::ZeroGhost};
}

} // namespace mpmstd::boundary
