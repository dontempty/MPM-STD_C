// Unit test: ScalarField / VectorField allocation and host access.

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
  auto& U = reg.add_vector("U");

  // Element count includes halos: (4+2)^3 = 216.
  MPMSTD_TEST_CHECK(T.n_elements() == 216);
  MPMSTD_TEST_CHECK(U.x().n_elements() == 216);

  // Fill, then check.
  T.fill_host(3.5);
  for (std::size_t i = 0; i < T.n_elements(); ++i) {
    MPMSTD_TEST_NEAR(T.host_ptr()[i], 3.5, 0.0);
  }

  // Components are independent.
  U.x().fill_host(1.0);
  U.y().fill_host(2.0);
  U.z().fill_host(3.0);
  MPMSTD_TEST_NEAR(U.x().host_ptr()[0], 1.0, 0.0);
  MPMSTD_TEST_NEAR(U.y().host_ptr()[0], 2.0, 0.0);
  MPMSTD_TEST_NEAR(U.z().host_ptr()[0], 3.0, 0.0);

  // host/device sync is a no-op on CPU build but must not crash.
  T.to_device();
  T.to_host();

  // Component lookup by enum.
  MPMSTD_TEST_NEAR(U.component(Component::U).host_ptr()[0], 1.0, 0.0);
  MPMSTD_TEST_NEAR(U.component(Component::W).host_ptr()[0], 3.0, 0.0);

  mpmstd_test_pass("field_alloc_and_access");
  std::cout << "test_field: ALL PASS\n";
  return 0;
}
