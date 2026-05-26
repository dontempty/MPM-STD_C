// Unit test: DomainTopology::sweep_order() returns the expected order for
// several wall-axis configurations.

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "test_helpers.hpp"

using namespace mpmstd;
using namespace mpmstd::boundary;

namespace {
void expect_order(DomainTopology& t,
                   Direction a0, Direction a1, Direction a2,
                   const char* name) {
  auto o = t.sweep_order();
  MPMSTD_TEST_CHECK(o[0] == a0);
  MPMSTD_TEST_CHECK(o[1] == a1);
  MPMSTD_TEST_CHECK(o[2] == a2);
  mpmstd_test_pass(name);
}
} // namespace

int main(int /*argc*/, char** /*argv*/) {
  // RBC default: z = wall  →  X, Y, Z
  {
    DomainTopology t;  // default constructor: x,y periodic, z non-periodic
    expect_order(t, Direction::X, Direction::Y, Direction::Z, "sweep_order_rbc_z_wall");
  }
  // PaScaL_TCS convention: y = wall  →  X, Z, Y
  {
    DomainTopology t;
    t.axis[to_int(Direction::X)] = AxisTopology::Periodic;
    t.axis[to_int(Direction::Y)] = AxisTopology::NonPeriodic;
    t.axis[to_int(Direction::Z)] = AxisTopology::Periodic;
    expect_order(t, Direction::X, Direction::Z, Direction::Y, "sweep_order_y_wall");
  }
  // x = wall  →  Y, Z, X
  {
    DomainTopology t;
    t.axis[to_int(Direction::X)] = AxisTopology::NonPeriodic;
    t.axis[to_int(Direction::Y)] = AxisTopology::Periodic;
    t.axis[to_int(Direction::Z)] = AxisTopology::Periodic;
    expect_order(t, Direction::Y, Direction::Z, Direction::X, "sweep_order_x_wall");
  }
  // Two walls (e.g. cavity): periodic axes first, walls last.  We don't
  // mandate the relative order of the two walls here; just check that
  // both walls are at positions [1], [2] and the periodic axis is at [0].
  {
    DomainTopology t;
    t.axis[to_int(Direction::X)] = AxisTopology::NonPeriodic;
    t.axis[to_int(Direction::Y)] = AxisTopology::Periodic;
    t.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;
    auto o = t.sweep_order();
    MPMSTD_TEST_CHECK(o[0] == Direction::Y);
    // remaining two can be (X, Z) or (Z, X); we accept the natural-order one
    MPMSTD_TEST_CHECK(o[1] == Direction::X && o[2] == Direction::Z);
    MPMSTD_TEST_CHECK(!t.wall_axis().has_value());   // ambiguous
    mpmstd_test_pass("sweep_order_two_walls");
  }
  std::cout << "test_sweep_order: ALL PASS\n";
  return 0;
}
