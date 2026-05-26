#include "boundary/domain_topology.hpp"

namespace mpmstd::boundary {

std::optional<Direction> DomainTopology::wall_axis() const {
  std::optional<Direction> found;
  for (int a = 0; a < 3; ++a) {
    if (axis[a] == AxisTopology::NonPeriodic) {
      if (found.has_value()) {
        // More than one non-periodic axis: caller must handle ambiguity.
        return std::nullopt;
      }
      found = static_cast<Direction>(a);
    }
  }
  return found;
}

std::array<Direction, 3> DomainTopology::sweep_order() const {
  std::array<Direction, 3> order{Direction::X, Direction::Y, Direction::Z};
  int k = 0;
  // First pass: periodic axes (any order — we use natural axis order).
  for (int a = 0; a < 3; ++a) {
    if (axis[a] == AxisTopology::Periodic) {
      order[k++] = static_cast<Direction>(a);
    }
  }
  // Second pass: non-periodic axes.
  for (int a = 0; a < 3; ++a) {
    if (axis[a] == AxisTopology::NonPeriodic) {
      order[k++] = static_cast<Direction>(a);
    }
  }
  return order;
}

} // namespace mpmstd::boundary
