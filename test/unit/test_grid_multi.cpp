// Multi-rank unit test: Grid metric arrays distribute correctly across ranks.
//
// We compare each rank's local slice of `xc` and `dx` against the global
// reference computed by `make_face_coordinates(...)`.  Run with several
// configurations to cover both uniform and tanh stretching, and both
// periodic and non-periodic axes.
//
// Run with: mpirun -np 8 test_grid_multi   (also accepts -np 1, 4)

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "grid/main.hpp"
#include "test_helpers.hpp"

#include <array>
#include <cmath>
#include <vector>

using namespace mpmstd;

namespace {

// Pick a topology that uses the world size exactly. Falls back to {1,1,1}
// on np=1.
std::array<int, 3> pick_dims(int world_size) {
  switch (world_size) {
    case 1: return {1, 1, 1};
    case 2: return {1, 1, 2};
    case 4: return {2, 2, 1};
    case 8: return {2, 2, 2};
  }
  throw std::runtime_error("test_grid_multi: unsupported world_size");
}

// Run one configuration and verify each rank's slice matches the global ref.
void verify_one_axis(const parallel::mpi::MpiContext& mpi,
                      std::array<int, 3> dims,
                      std::array<bool, 3> periodic,
                      std::array<int, 3>   n_global,
                      std::array<grid::AxisConfig, 3> axes,
                      const char* tag) {
  parallel::mpi::MpiTopology topo(mpi, dims, periodic);
  parallel::mpi::Subdomain   sub (topo, n_global);
  grid::Grid g(sub, axes);

  for (int a = 0; a < 3; ++a) {
    const Direction d = static_cast<Direction>(a);
    const int n_glb = axes[a].n_global;
    auto global_faces = grid::make_face_coordinates(
        axes[a].stretch, n_glb, axes[a].length, axes[a].gamma);

    // Verify local cell-width dx matches the global slice (interior only).
    const auto& dx_local = g.dx(d);
    const int   n_int    = sub.n_interior()[a];
    const int   off      = sub.global_offset()[a];

    for (int i = 0; i < n_int; ++i) {
      const real_t expected = global_faces[off + i + 1] - global_faces[off + i];
      const real_t got      = dx_local[i + kHaloWidth];
      MPMSTD_TEST_NEAR(got, expected, 1e-13);
    }

    // Verify local cell-center xc matches the global slice (interior only).
    const auto& xc_local = g.xc(d);
    for (int i = 0; i < n_int; ++i) {
      const real_t expected = 0.5 * (global_faces[off + i] + global_faces[off + i + 1]);
      const real_t got      = xc_local[i + kHaloWidth];
      MPMSTD_TEST_NEAR(got, expected, 1e-13);
    }
  }

  if (mpi.is_root()) mpmstd_test_pass(tag);
}

} // namespace

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  const auto dims = pick_dims(mpi.world_size());

  // (1) Uniform on all three axes, all periodic.
  verify_one_axis(mpi, dims, {true, true, true}, {8, 8, 8},
                   {{
                     {8, 1.0, grid::StretchKind::Uniform, 0.0},
                     {8, 1.0, grid::StretchKind::Uniform, 0.0},
                     {8, 1.0, grid::StretchKind::Uniform, 0.0},
                   }}, "grid_multi_uniform_periodic");

  // (2) Uniform on x,y, tanh on z, z = wall (non-periodic).
  verify_one_axis(mpi, dims, {true, true, false}, {16, 16, 16},
                   {{
                     {16, 4.0, grid::StretchKind::Uniform, 0.0},
                     {16, 2.0, grid::StretchKind::Uniform, 0.0},
                     {16, 2.0, grid::StretchKind::Tanh,    2.6},
                   }}, "grid_multi_uniform_xy_tanh_z");

  if (mpi.is_root()) std::cout << "test_grid_multi: ALL PASS\n";
  return 0;
}
