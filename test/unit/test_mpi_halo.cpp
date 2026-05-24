// Unit test: halo exchange on an 8-rank Cartesian topology.
//
// Run with: mpirun -np 8 test_mpi_halo
//
// Each interior cell is initialised with a globally-unique value f(gi, gj, gk).
// After exchange_halo, the halo cells must equal the same f evaluated at the
// neighbouring rank's interior position.

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace mpmstd;

namespace {

inline real_t encode(int gi, int gj, int gk) {
  return static_cast<real_t>(gi) * 1000.0 +
         static_cast<real_t>(gj) * 1.0    +
         static_cast<real_t>(gk) * 0.001;
}

} // namespace

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  MPMSTD_TEST_CHECK(mpi.world_size() == 8);

  parallel::mpi::MpiTopology topo(mpi, {2, 2, 2}, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {8, 8, 8});   // 4 interior cells per axis per rank

  auto backend = parallel::make_default_backend();
  field::FieldRegistry reg(sub, *backend);
  auto& T = reg.add_scalar("T");

  // Initialise interior cells with global-coordinate-encoded value.
  const auto n_int = sub.n_interior();
  const auto off   = sub.global_offset();
  for (int i = 0; i < n_int[0]; ++i)
    for (int j = 0; j < n_int[1]; ++j)
      for (int k = 0; k < n_int[2]; ++k) {
        const int li = i + kHaloWidth;
        const int lj = j + kHaloWidth;
        const int lk = k + kHaloWidth;
        T.host_at(li, lj, lk) = encode(off[0] + i, off[1] + j, off[2] + k);
      }

  T.exchange_halo();

  // Check the 6 face-halo planes by sampling the central row of each.
  // For a 4-interior, halo=1 layout: lower halo at idx 0, upper halo at idx 5.
  const int ihi = sub.n_total()[0] - 1;
  const int jhi = sub.n_total()[1] - 1;
  const int khi = sub.n_total()[2] - 1;
  const int ng_x = 8, ng_y = 8, ng_z = 8;

  // Negative-x halo: should contain encode((off[0] - 1 + ng_x) % ng_x, off[1]+j, off[2]+k)
  for (int j = 0; j < n_int[1]; ++j) {
    for (int k = 0; k < n_int[2]; ++k) {
      const int lj = j + kHaloWidth;
      const int lk = k + kHaloWidth;
      const real_t v = T.host_at(0, lj, lk);
      const int gi  = (off[0] - 1 + ng_x) % ng_x;
      const real_t e = encode(gi, off[1] + j, off[2] + k);
      MPMSTD_TEST_NEAR(v, e, 1e-12);
    }
  }
  // Positive-x halo
  for (int j = 0; j < n_int[1]; ++j) {
    for (int k = 0; k < n_int[2]; ++k) {
      const int lj = j + kHaloWidth;
      const int lk = k + kHaloWidth;
      const real_t v = T.host_at(ihi, lj, lk);
      const int gi  = (off[0] + n_int[0]) % ng_x;
      const real_t e = encode(gi, off[1] + j, off[2] + k);
      MPMSTD_TEST_NEAR(v, e, 1e-12);
    }
  }
  // Negative-y halo
  for (int i = 0; i < n_int[0]; ++i) {
    for (int k = 0; k < n_int[2]; ++k) {
      const int li = i + kHaloWidth;
      const int lk = k + kHaloWidth;
      const real_t v = T.host_at(li, 0, lk);
      const int gj  = (off[1] - 1 + ng_y) % ng_y;
      const real_t e = encode(off[0] + i, gj, off[2] + k);
      MPMSTD_TEST_NEAR(v, e, 1e-12);
    }
  }
  // Positive-y halo
  for (int i = 0; i < n_int[0]; ++i) {
    for (int k = 0; k < n_int[2]; ++k) {
      const int li = i + kHaloWidth;
      const int lk = k + kHaloWidth;
      const real_t v = T.host_at(li, jhi, lk);
      const int gj  = (off[1] + n_int[1]) % ng_y;
      const real_t e = encode(off[0] + i, gj, off[2] + k);
      MPMSTD_TEST_NEAR(v, e, 1e-12);
    }
  }
  // Negative-z halo
  for (int i = 0; i < n_int[0]; ++i) {
    for (int j = 0; j < n_int[1]; ++j) {
      const int li = i + kHaloWidth;
      const int lj = j + kHaloWidth;
      const real_t v = T.host_at(li, lj, 0);
      const int gk  = (off[2] - 1 + ng_z) % ng_z;
      const real_t e = encode(off[0] + i, off[1] + j, gk);
      MPMSTD_TEST_NEAR(v, e, 1e-12);
    }
  }
  // Positive-z halo
  for (int i = 0; i < n_int[0]; ++i) {
    for (int j = 0; j < n_int[1]; ++j) {
      const int li = i + kHaloWidth;
      const int lj = j + kHaloWidth;
      const real_t v = T.host_at(li, lj, khi);
      const int gk  = (off[2] + n_int[2]) % ng_z;
      const real_t e = encode(off[0] + i, off[1] + j, gk);
      MPMSTD_TEST_NEAR(v, e, 1e-12);
    }
  }

  if (mpi.is_root()) {
    mpmstd_test_pass("mpi_halo_exchange_periodic_2x2x2");
    std::cout << "test_mpi_halo: ALL PASS\n";
  }
  return 0;
}
