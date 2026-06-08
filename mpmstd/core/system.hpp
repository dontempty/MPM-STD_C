#pragma once

#include "core/bands.hpp"
#include "common/direction.hpp"   // Direction, Component, to_int

#include <array>

namespace mpmstd::core {

// =============================================================================
// *System  — the Ax=b bundle for one equation. assemble_* fills it; solve_*
//            consumes it. One per equation type (rev.2 §4). For the spike we
//            pin the SHAPE / signature; full per-field content lands in P1.
// =============================================================================

// One scalar transport equation: a Bands per sweep direction (3-stage ADI).
struct ScalarSystem {
  Bands x, y, z;

  Bands& along(Direction d) {
    return d == Direction::X ? x : (d == Direction::Y ? y : z);
  }
  const Bands& along(Direction d) const {
    return d == Direction::X ? x : (d == Direction::Y ? y : z);
  }
};

// Momentum: 3 velocity components, each a 3-direction ADI system. The block
// lower-triangular velocity coupling (blockLdV/blockLdU in the MPM-STD fortran
// core_momentum) is handled INSIDE solve_momentum_* (rev.2 M2), not as a
// separate call in main — so MomentumSystem just carries the per-component
// directional Bands here; coupling metadata is added in P1.
struct MomentumSystem {
  ScalarSystem comp[3];   // index by Component: U=0, V=1, W=2

  ScalarSystem&       component(Component c)       { return comp[to_int(c)]; }
  const ScalarSystem& component(Component c) const { return comp[to_int(c)]; }
};

// Per-axis spectral transform used by the pressure Poisson solve. Chosen from
// the BC (rev.2 §9c): periodic -> Fft, Neumann/wall on a uniform axis -> Dct,
// the remaining axis -> Tdma.
enum class Transform { Fft, Dct, Tdma };

// Pressure-Poisson system. The wavenumber tables + transform plans + RHS buffer
// are filled when the Poisson solver is ported (P3/P4); the spike only pins the
// type's existence and the per-axis transform tag so the API is frozen.
struct PressureSystem {
  std::array<Transform, 3> transform{Transform::Tdma, Transform::Tdma, Transform::Tdma};
};

} // namespace mpmstd::core
