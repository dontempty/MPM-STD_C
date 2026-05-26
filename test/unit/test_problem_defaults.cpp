// Unit test: Problem default constructor yields the neutral all-Periodic state.
//
// All non-trivial problem setups (RBC, channel, cavity, ...) come from
// load_problem_from_config — the library deliberately ships *no* physics-
// specific defaults.

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "test_helpers.hpp"

using namespace mpmstd;
using namespace mpmstd::boundary;

int main(int /*argc*/, char** /*argv*/) {
  Problem p;

  // Topology: all axes periodic (the only consistent neutral default).
  for (Direction d : { Direction::X, Direction::Y, Direction::Z }) {
    MPMSTD_TEST_CHECK(p.topology.is_periodic(d));
  }
  // No wall axis when everything is periodic.
  MPMSTD_TEST_CHECK(!p.topology.wall_axis().has_value());

  // Every face of every field is Periodic.
  const FieldBoundary* fbs[5] = { &p.U, &p.V, &p.W, &p.P, &p.T };
  for (const FieldBoundary* fb : fbs) {
    for (Direction d : { Direction::X, Direction::Y, Direction::Z }) {
      for (Side s : { Side::Minus, Side::Plus }) {
        MPMSTD_TEST_CHECK(fb->face(d, s).kind == BcKind::Periodic);
      }
    }
  }

  // validate() must pass on a fresh Problem.
  p.validate();

  // apply_periodic_defaults() is idempotent: calling it again leaves the
  // state unchanged.
  p.apply_periodic_defaults();
  p.validate();
  MPMSTD_TEST_CHECK(p.U.face(Direction::X, Side::Minus).kind == BcKind::Periodic);

  mpmstd_test_pass("problem_default_is_neutral_periodic");
  std::cout << "test_problem_defaults: ALL PASS\n";
  return 0;
}
