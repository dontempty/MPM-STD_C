// Unit test: ScalarField allocation and host access, including the case
// where multiple scalars (e.g. velocity components U, V, W) are registered
// side-by-side in a FieldRegistry — the convention we follow now that
// VectorField has been removed.

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "test_helpers.hpp"

using namespace mpmstd;

int main(int /*argc*/, char** /*argv*/) {
  parallel::mpi::MpiContext   mpi(nullptr, nullptr);
  parallel::mpi::MpiTopology  topo(mpi, {1, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain    sub(topo, {4, 4, 4});

  auto backend = parallel::make_default_backend();

  field::FieldRegistry reg(sub, *backend);
  auto& T = reg.add_scalar("T");

  // PaScaL_TCS-style: each velocity component is its own ScalarField.
  auto& U = reg.add_scalar("U");
  auto& V = reg.add_scalar("V");
  auto& W = reg.add_scalar("W");

  // Element count includes halos: (4+2)^3 = 216.
  MPMSTD_TEST_CHECK(T.n_elements() == 216);
  MPMSTD_TEST_CHECK(U.n_elements() == 216);

  // Fill, then check.
  T.fill_host(3.5);
  for (std::size_t i = 0; i < T.n_elements(); ++i) {
    MPMSTD_TEST_NEAR(T.host_ptr()[i], 3.5, 0.0);
  }

  // Components are independent — three separate allocations behind the scenes.
  U.fill_host(1.0);
  V.fill_host(2.0);
  W.fill_host(3.0);
  MPMSTD_TEST_NEAR(U.host_ptr()[0], 1.0, 0.0);
  MPMSTD_TEST_NEAR(V.host_ptr()[0], 2.0, 0.0);
  MPMSTD_TEST_NEAR(W.host_ptr()[0], 3.0, 0.0);

  // host/device sync is a no-op on CPU build but must not crash.
  T.to_device();
  T.to_host();

  // Registry lookup by name.
  MPMSTD_TEST_NEAR(reg.scalar("U").host_ptr()[0], 1.0, 0.0);
  MPMSTD_TEST_NEAR(reg.scalar("W").host_ptr()[0], 3.0, 0.0);

  // Adding the same name twice throws.
  bool threw = false;
  try { reg.add_scalar("T"); } catch (const std::runtime_error&) { threw = true; }
  MPMSTD_TEST_CHECK(threw);

  mpmstd_test_pass("field_alloc_and_access");
  std::cout << "test_field: ALL PASS\n";
  return 0;
}
