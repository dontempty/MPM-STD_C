// Unit test: grid stretching and metric arrays.
//
// Single-rank: we build a 1x1x1 process topology so a Subdomain can be
// constructed without needing mpirun.

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "grid/main.hpp"
#include "test_helpers.hpp"

using namespace mpmstd;

static void test_uniform_grid_single_rank(const parallel::mpi::MpiContext& mpi) {
  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain   sub(topo, {8, 8, 8});

  grid::Grid g(sub, {{
    { /*n_global=*/8, /*length=*/1.0, grid::StretchKind::Uniform, 0.0 },
    { 8,             1.0,             grid::StretchKind::Uniform, 0.0 },
    { 8,             1.0,             grid::StretchKind::Uniform, 0.0 },
  }});

  // Uniform: all dx[i] should equal 1/8 for interior cells.
  for (int i = kHaloWidth; i < kHaloWidth + 8; ++i) {
    MPMSTD_TEST_NEAR(g.dx(Direction::X)[i], 0.125, 1e-14);
    MPMSTD_TEST_NEAR(g.dx(Direction::Y)[i], 0.125, 1e-14);
    MPMSTD_TEST_NEAR(g.dx(Direction::Z)[i], 0.125, 1e-14);
  }
  // Cell centers at 1/16, 3/16, 5/16, ...
  for (int i = 0; i < 8; ++i) {
    const real_t expected = (2.0 * i + 1.0) / 16.0;
    MPMSTD_TEST_NEAR(g.xc(Direction::X)[i + kHaloWidth], expected, 1e-14);
  }
  mpmstd_test_pass("uniform_grid_single_rank");
}

static void test_tanh_stretching_endpoints(const parallel::mpi::MpiContext& mpi) {
  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, false, true});
  parallel::mpi::Subdomain   sub(topo, {16, 16, 16});

  grid::Grid g(sub, {{
    { 16, 4.0, grid::StretchKind::Uniform, 0.0 },
    { 16, 2.0, grid::StretchKind::Tanh,    2.6 },   // y-axis wall-clustered
    { 16, 2.0, grid::StretchKind::Uniform, 0.0 },
  }});

  // First interior face should be exactly 0 (lower wall).
  // (Recall: xf has size n_total+1; interior face k=kHaloWidth corresponds to global face 0.)
  MPMSTD_TEST_NEAR(g.xf(Direction::Y)[kHaloWidth],        0.0, 1e-14);
  MPMSTD_TEST_NEAR(g.xf(Direction::Y)[kHaloWidth + 16],   2.0, 1e-14);

  // dx should be smaller near the walls than in the middle (clustering).
  const auto& dy = g.dx(Direction::Y);
  const real_t dy_wall   = dy[kHaloWidth];          // first interior cell
  const real_t dy_centre = dy[kHaloWidth + 8];      // middle cell
  MPMSTD_TEST_CHECK(dy_wall < dy_centre);

  // Symmetric: dy at index 0 (lower) == dy at index 15 (upper) interior.
  MPMSTD_TEST_NEAR(dy[kHaloWidth], dy[kHaloWidth + 15], 1e-13);

  // Sum of interior dx[i] should equal L (=2.0) within FP tol.
  real_t s = 0.0;
  for (int i = kHaloWidth; i < kHaloWidth + 16; ++i) s += dy[i];
  MPMSTD_TEST_NEAR(s, 2.0, 1e-13);

  mpmstd_test_pass("tanh_stretching_endpoints");
}

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  test_uniform_grid_single_rank(mpi);
  test_tanh_stretching_endpoints(mpi);
  if (mpi.is_root()) std::cout << "test_grid: ALL PASS\n";
  return 0;
}
