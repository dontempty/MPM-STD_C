#include "boundary/face_bc.hpp"

namespace mpmstd::boundary {

BcValueFn FaceBc::constant(real_t v) {
  return [v](real_t, real_t, real_t, real_t) { return v; };
}

FaceBc FaceBc::periodic() {
  return FaceBc{BcKind::Periodic, constant(0.0)};
}
FaceBc FaceBc::dirichlet(real_t v) {
  return FaceBc{BcKind::Dirichlet, constant(v), GhostPolicy::ZeroGhost};
}
FaceBc FaceBc::dirichlet(BcValueFn f) {
  return FaceBc{BcKind::Dirichlet, std::move(f), GhostPolicy::ZeroGhost};
}
FaceBc FaceBc::dirichlet_antisymm(real_t v) {
  return FaceBc{BcKind::Dirichlet, constant(v), GhostPolicy::Antisymmetric};
}
FaceBc FaceBc::dirichlet_antisymm(BcValueFn f) {
  return FaceBc{BcKind::Dirichlet, std::move(f), GhostPolicy::Antisymmetric};
}
FaceBc FaceBc::neumann(real_t v) {
  return FaceBc{BcKind::Neumann, constant(v)};
}
FaceBc FaceBc::neumann(BcValueFn f) {
  return FaceBc{BcKind::Neumann, std::move(f)};
}

} // namespace mpmstd::boundary
