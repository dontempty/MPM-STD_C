#pragma once

namespace mpmstd::boundary {

// =============================================================================
// Boundary-condition kind (per face per field).
//
// Phase-1 implements only Periodic, Dirichlet, Neumann.  The remaining enum
// values reserve slots for future BC types so that adding them later requires
// only filling in `BoundaryApplier::apply_ghost` and `modify_tdma_row` cases.
// Calling those cases before they are implemented throws a runtime_error.
// =============================================================================
enum class BcKind {
  // === 1차 구현 ===
  Periodic   = 0,
  Dirichlet  = 1,
  Neumann    = 2,

  // === 미래 확장 (인터페이스만, 1차에서는 throw stub) ===
  Wall       = 100,
  Inflow     = 101,
  Outflow    = 102,
};

constexpr const char* bc_kind_name(BcKind k) {
  switch (k) {
    case BcKind::Periodic:  return "Periodic";
    case BcKind::Dirichlet: return "Dirichlet";
    case BcKind::Neumann:   return "Neumann";
    case BcKind::Wall:      return "Wall";
    case BcKind::Inflow:    return "Inflow";
    case BcKind::Outflow:   return "Outflow";
  }
  return "?";
}

constexpr bool bc_kind_is_implemented(BcKind k) {
  return k == BcKind::Periodic
      || k == BcKind::Dirichlet
      || k == BcKind::Neumann;
}

} // namespace mpmstd::boundary
