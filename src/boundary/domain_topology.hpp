#pragma once

#include "common/direction.hpp"

#include <array>
#include <optional>

namespace mpmstd::boundary {

// Per-axis topology (which axes are periodic and which are non-periodic).
enum class AxisTopology { Periodic, NonPeriodic };

// DomainTopology carries the periodicity of each of the three Cartesian axes,
// and provides the derived helpers that the rest of the codebase uses
// (sweep_order, wall_axis, is_periodic).

struct DomainTopology {
  std::array<AxisTopology, 3> axis{
    AxisTopology::Periodic,     // x  (default RBC: streamwise / lateral)
    AxisTopology::Periodic,     // y  (default RBC: lateral)
    AxisTopology::NonPeriodic   // z  (default RBC: wall-normal)
  };

  // ----- query helpers -----
  bool is_periodic(Direction d) const {
    return axis[to_int(d)] == AxisTopology::Periodic;
  }
  bool is_non_periodic(Direction d) const { return !is_periodic(d); }

  // Returns the (unique) non-periodic axis if there is exactly one; otherwise
  // returns std::nullopt.  Phase-1 always uses a single wall axis, so callers
  // that do .value() implicitly enforce that invariant.
  std::optional<Direction> wall_axis() const;

  // ADI sweep order: periodic axes first, the wall (non-periodic) axis last.
  // For RBC default (z = wall) this returns (X, Y, Z).
  // For PaScaL_TCS convention (y = wall) it would return (X, Z, Y).
  std::array<Direction, 3> sweep_order() const;
};

} // namespace mpmstd::boundary
