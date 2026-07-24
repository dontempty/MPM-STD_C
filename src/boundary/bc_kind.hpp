#pragma once

namespace mpmstd::boundary {

// =============================================================================
// Boundary-condition kind (per face per field): Periodic, Dirichlet, Neumann.
// =============================================================================
enum class BcKind {
  Periodic   = 0,
  Dirichlet  = 1,
  Neumann    = 2,
};

constexpr const char* bc_kind_name(BcKind k) {
  switch (k) {
    case BcKind::Periodic:  return "Periodic";
    case BcKind::Dirichlet: return "Dirichlet";
    case BcKind::Neumann:   return "Neumann";
  }
  return "?";
}

constexpr bool bc_kind_is_implemented(BcKind k) {
  return k == BcKind::Periodic
      || k == BcKind::Dirichlet
      || k == BcKind::Neumann;
}

} // namespace mpmstd::boundary
