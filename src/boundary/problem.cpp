#include "boundary/problem.hpp"

#include <stdexcept>
#include <string>

namespace mpmstd::boundary {

Problem::Problem() {
  apply_periodic_defaults();
}

void Problem::apply_periodic_defaults() {
  // Neutral default: every axis periodic, every field face Periodic.
  // Non-trivial problems are populated by load_problem_from_config() or by
  // explicit per-face overrides — never by hidden physics-specific defaults.
  for (int a = 0; a < 3; ++a) {
    topology.axis[a] = AxisTopology::Periodic;
  }

  FieldBoundary* fields[5] = { &U, &V, &W, &P, &T };
  for (FieldBoundary* fb : fields) {
    for (Direction d : { Direction::X, Direction::Y, Direction::Z }) {
      for (Side s : { Side::Minus, Side::Plus }) {
        fb->face(d, s) = FaceBc::periodic();
      }
    }
  }
}

void Problem::validate() const {
  // Invariant: for every field, if topology.axis[d] is Periodic, then both
  // faces along d must have BcKind::Periodic — and vice versa.
  const FieldBoundary* fields[5] = { &U, &V, &W, &P, &T };
  const char* names[5]           = { "U", "V", "W", "P", "T" };

  for (int f = 0; f < 5; ++f) {
    for (int a = 0; a < 3; ++a) {
      const Direction d   = static_cast<Direction>(a);
      const bool axis_per = topology.is_periodic(d);
      for (Side s : { Side::Minus, Side::Plus }) {
        const BcKind k = fields[f]->face(d, s).kind;
        const bool face_per = (k == BcKind::Periodic);
        if (axis_per != face_per) {
          throw std::runtime_error(
            std::string("Problem::validate: field '") + names[f] +
            "' face (" + direction_name(d) + "," + side_name(s) +
            ") has BcKind=" + bc_kind_name(k) +
            " but axis " + direction_name(d) +
            (axis_per ? " is Periodic" : " is NonPeriodic"));
        }
      }
    }
  }
}

} // namespace mpmstd::boundary
