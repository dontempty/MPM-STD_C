// Unit test: closed cavity configuration (all 6 faces non-periodic, all
// fields Dirichlet on every face).
//
// Verifies that the boundary subsystem handles a multi-wall topology without
// any special-casing:
//   * Problem::validate() passes when every axis is NonPeriodic and every
//     face is a non-Periodic kind.
//   * DomainTopology::wall_axis() returns nullopt for multi-wall (ambiguous),
//     and sweep_order() returns a deterministic order.
//   * BoundaryApplier::apply_ghost() fills all 6 ghost planes with the
//     correct Dirichlet values on a single-rank domain.

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "field/main.hpp"
#include "parallel/main.hpp"
#include "test_helpers.hpp"

#include <array>

using namespace mpmstd;
using namespace mpmstd::boundary;

namespace {

// Helper: apply Dirichlet `val` to every face of every field in the Problem.
void set_all_faces_dirichlet(Problem& p, real_t val_uvw, real_t val_p_neumann,
                              std::array<real_t, 6> T_face_values) {
  for (auto* fb : { &p.U, &p.V, &p.W }) {
    for (Direction d : { Direction::X, Direction::Y, Direction::Z }) {
      for (Side s : { Side::Minus, Side::Plus }) {
        fb->face(d, s) = FaceBc::dirichlet(val_uvw);
      }
    }
  }
  for (Direction d : { Direction::X, Direction::Y, Direction::Z }) {
    for (Side s : { Side::Minus, Side::Plus }) {
      p.P.face(d, s) = FaceBc::neumann(val_p_neumann);
    }
  }
  // T: per-face values for distinguishability.
  // Layout: [-x, +x, -y, +y, -z, +z]
  p.T.face(Direction::X, Side::Minus) = FaceBc::dirichlet(T_face_values[0]);
  p.T.face(Direction::X, Side::Plus ) = FaceBc::dirichlet(T_face_values[1]);
  p.T.face(Direction::Y, Side::Minus) = FaceBc::dirichlet(T_face_values[2]);
  p.T.face(Direction::Y, Side::Plus ) = FaceBc::dirichlet(T_face_values[3]);
  p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet(T_face_values[4]);
  p.T.face(Direction::Z, Side::Plus ) = FaceBc::dirichlet(T_face_values[5]);
}

} // anonymous namespace

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // -----------------------------------------------------------------------
  // Section 1: pure-topology checks (no MPI/Field involvement).
  // -----------------------------------------------------------------------
  {
    Problem p;
    // All three axes non-periodic (the default is now all-Periodic).
    p.topology.axis[to_int(Direction::X)] = AxisTopology::NonPeriodic;
    p.topology.axis[to_int(Direction::Y)] = AxisTopology::NonPeriodic;
    p.topology.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;

    // Distinct per-face T values let the apply_ghost test below distinguish
    // which face ended up where.
    set_all_faces_dirichlet(p, /*velocity wall*/ 0.0, /*pressure Neumann*/ 0.0,
                              /*T faces*/ {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    // (a) validate() must pass — every face kind is non-Periodic and every
    //     axis is NonPeriodic, so the invariant is satisfied.
    p.validate();
    mpmstd_test_pass("cavity_problem_validate_passes");

    // (b) wall_axis() returns nullopt because there are multiple walls.
    MPMSTD_TEST_CHECK(!p.topology.wall_axis().has_value());
    mpmstd_test_pass("cavity_wall_axis_is_ambiguous");

    // (c) sweep_order(): with all axes NonPeriodic, periodic loop runs zero
    //     times and the non-periodic loop fills in natural axis order
    //     (X, Y, Z).
    auto order = p.topology.sweep_order();
    MPMSTD_TEST_CHECK(order[0] == Direction::X);
    MPMSTD_TEST_CHECK(order[1] == Direction::Y);
    MPMSTD_TEST_CHECK(order[2] == Direction::Z);
    mpmstd_test_pass("cavity_sweep_order_all_walls");
  }

  // -----------------------------------------------------------------------
  // Section 2: apply_ghost on a single-rank domain — this rank owns every
  // global boundary face, so all 6 ghost planes should get the prescribed
  // Dirichlet values.
  // -----------------------------------------------------------------------
  {
    parallel::mpi::MpiTopology topo(mpi, {1, 1, 1},
                                     {false, false, false});   // every axis non-periodic
    parallel::mpi::Subdomain   sub (topo, {4, 4, 4});           // tiny grid

    auto backend = parallel::make_default_backend();
    field::FieldRegistry reg(sub, *backend);
    auto& T = reg.add_scalar("T");

    constexpr real_t SENTINEL = -999.0;
    T.fill_host(SENTINEL);
    // Fill interior with something distinct (we only care that ghosts get
    // overwritten, but a non-sentinel interior helps confirm we don't
    // accidentally touch interior cells either).
    const auto n_tot = sub.n_total();
    for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j)
        for (int k = kHaloWidth; k < n_tot[2] - kHaloWidth; ++k)
          T.host_at(i, j, k) = 0.0;

    // Same Problem setup as Section 1: 6 distinct T face values.
    Problem p;
    p.topology.axis[to_int(Direction::X)] = AxisTopology::NonPeriodic;
    p.topology.axis[to_int(Direction::Y)] = AxisTopology::NonPeriodic;
    set_all_faces_dirichlet(p, 0.0, 0.0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    BoundaryApplier app(p);
    app.apply_ghost(T, p.T);

    // ----- check every ghost plane was overwritten with the correct value -----
    const int n0 = n_tot[0], n1 = n_tot[1], n2 = n_tot[2];

    // -x ghost (i = 0): expected 1.0
    for (int j = kHaloWidth; j < n1 - kHaloWidth; ++j)
      for (int k = kHaloWidth; k < n2 - kHaloWidth; ++k)
        MPMSTD_TEST_NEAR(T.host_at(0, j, k), 1.0, 0.0);
    // +x ghost (i = n0-1): expected 2.0
    for (int j = kHaloWidth; j < n1 - kHaloWidth; ++j)
      for (int k = kHaloWidth; k < n2 - kHaloWidth; ++k)
        MPMSTD_TEST_NEAR(T.host_at(n0 - 1, j, k), 2.0, 0.0);

    // -y ghost (j = 0): expected 3.0
    for (int i = kHaloWidth; i < n0 - kHaloWidth; ++i)
      for (int k = kHaloWidth; k < n2 - kHaloWidth; ++k)
        MPMSTD_TEST_NEAR(T.host_at(i, 0, k), 3.0, 0.0);
    // +y ghost (j = n1-1): expected 4.0
    for (int i = kHaloWidth; i < n0 - kHaloWidth; ++i)
      for (int k = kHaloWidth; k < n2 - kHaloWidth; ++k)
        MPMSTD_TEST_NEAR(T.host_at(i, n1 - 1, k), 4.0, 0.0);

    // -z ghost (k = 0): expected 5.0
    for (int i = kHaloWidth; i < n0 - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n1 - kHaloWidth; ++j)
        MPMSTD_TEST_NEAR(T.host_at(i, j, 0), 5.0, 0.0);
    // +z ghost (k = n2-1): expected 6.0
    for (int i = kHaloWidth; i < n0 - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n1 - kHaloWidth; ++j)
        MPMSTD_TEST_NEAR(T.host_at(i, j, n2 - 1), 6.0, 0.0);

    // Interior must remain at 0.0 (untouched by apply_ghost).
    for (int i = kHaloWidth; i < n0 - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n1 - kHaloWidth; ++j)
        for (int k = kHaloWidth; k < n2 - kHaloWidth; ++k)
          MPMSTD_TEST_NEAR(T.host_at(i, j, k), 0.0, 0.0);

    mpmstd_test_pass("cavity_apply_ghost_fills_all_six_walls");
  }

  // -----------------------------------------------------------------------
  // Section 3: validate() must REJECT the mismatched case where the user
  // forgot to update one axis topology after switching its face BC.
  // -----------------------------------------------------------------------
  {
    Problem p;
    // x stays Periodic (default) but the user sets T face on -x to Dirichlet:
    p.T.face(Direction::X, Side::Minus) = FaceBc::dirichlet(1.0);

    bool threw = false;
    try { p.validate(); } catch (const std::runtime_error&) { threw = true; }
    MPMSTD_TEST_CHECK(threw);
    mpmstd_test_pass("cavity_validate_rejects_topology_face_mismatch");
  }

  if (mpi.is_root()) std::cout << "test_problem_cavity: ALL PASS\n";
  return 0;
}
