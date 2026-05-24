// Integration test: write a ScalarField to disk, read it back into a different
// field, and verify bit-for-bit equality on the interior.
//
// Run with: mpirun -np 8 test_restart_roundtrip   (also accepts -np 1, 2, 4)

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "post/main.hpp"
#include "test_helpers.hpp"

#include <cstdio>

using namespace mpmstd;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // Topology adapts to whatever world size the user runs us with.
  std::array<int, 3> dims = {1, 1, 1};
  switch (mpi.world_size()) {
    case 1:  dims = {1, 1, 1}; break;
    case 2:  dims = {2, 1, 1}; break;
    case 4:  dims = {2, 2, 1}; break;
    case 8:  dims = {2, 2, 2}; break;
    default:
      if (mpi.is_root())
        std::fprintf(stderr, "test_restart_roundtrip: unsupported world size %d\n",
                      mpi.world_size());
      return 1;
  }

  parallel::mpi::MpiTopology topo(mpi, dims, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {8, 8, 8});

  auto backend = parallel::make_default_backend();

  field::FieldRegistry reg(sub, *backend);
  auto& T_write = reg.add_scalar("T_write");
  auto& T_read  = reg.add_scalar("T_read");

  // Fill interior with a globally-unique pattern.
  const auto n_int = sub.n_interior();
  const auto off   = sub.global_offset();
  for (int i = 0; i < n_int[0]; ++i)
    for (int j = 0; j < n_int[1]; ++j)
      for (int k = 0; k < n_int[2]; ++k) {
        const int li = i + kHaloWidth;
        const int lj = j + kHaloWidth;
        const int lk = k + kHaloWidth;
        T_write.host_at(li, lj, lk) =
            static_cast<real_t>((off[0] + i) * 100 + (off[1] + j) * 10 + (off[2] + k));
      }

  const std::string path = "/tmp/mpmstd_test_restart.bin";
  post::write_scalar(T_write, path);
  post::read_scalar (T_read,  path);

  // Compare interiors only — halos may differ (we did not exchange).
  for (int i = 0; i < n_int[0]; ++i)
    for (int j = 0; j < n_int[1]; ++j)
      for (int k = 0; k < n_int[2]; ++k) {
        const int li = i + kHaloWidth;
        const int lj = j + kHaloWidth;
        const int lk = k + kHaloWidth;
        MPMSTD_TEST_NEAR(T_read.host_at(li, lj, lk),
                          T_write.host_at(li, lj, lk), 0.0);
      }

  // Clean up the test file (root only).
  MPI_Barrier(topo.cart_comm());
  if (mpi.is_root()) {
    std::remove(path.c_str());
    mpmstd_test_pass("restart_write_read_roundtrip");
    std::cout << "test_restart_roundtrip: ALL PASS\n";
  }
  return 0;
}
