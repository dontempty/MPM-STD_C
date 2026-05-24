#pragma once

#include "common/types.hpp"
#include "common/direction.hpp"
#include "common/macros.hpp"
#include "parallel/backend/backend.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <string>
#include <vector>

namespace mpmstd::field {

// Cell-centered (or face-centered, depending on the variable's semantic) 3D
// field. All scalar / vector-component arrays in this codebase share the same
// shape — the staggered MAC convention is encoded by the **indexing rules**
// used inside stencil functions, not by the storage shape itself. This mirrors
// PaScaL_TCS, where `U(:,:,:)`, `V(:,:,:)`, `P(:,:,:)`, `T(:,:,:)` all have the
// same `(0:n1sub, 0:n2sub, 0:n3sub)` shape.
//
// Memory model
// ------------
//   Host buffer  : always allocated  (used for IO / restart / unit tests)
//   Device buffer: CUDA build only   (hot-loop kernels read/write this)
//
// On CPU build `device_ptr() == host_ptr()` and the to_*() routines are
// no-ops, so callers can use the same code path in both builds.

class ScalarField {
public:
  ScalarField(const parallel::mpi::Subdomain& sub,
              parallel::Backend& backend,
              std::string name);
  ~ScalarField();

  ScalarField(const ScalarField&) = delete;
  ScalarField& operator=(const ScalarField&) = delete;

  // ----- metadata -----
  const std::string& name()    const { return name_; }
  std::size_t        n_elements() const { return n_elements_; }
  int n_total(Direction d) const { return sub_.n_total(d); }
  int n_interior(Direction d) const { return sub_.n_interior(d); }
  const parallel::mpi::Subdomain& subdomain() const { return sub_; }

  // Row-major flat index.
  int linear_index(int i, int j, int k) const { return sub_.linear_index(i, j, k); }

  // ----- host access (always valid) -----
  real_t*       host_ptr()        { return host_.data(); }
  const real_t* host_ptr() const  { return host_.data(); }

  real_t&       host_at(int i, int j, int k)       { return host_[linear_index(i,j,k)]; }
  const real_t& host_at(int i, int j, int k) const { return host_[linear_index(i,j,k)]; }

  // Whole-array operations (host).
  void fill_host(real_t value);
  void copy_from_host(const real_t* src);   // src must have n_elements() entries

  // ----- device access -----
  // CPU build : returns host pointer (so callers can pass uniformly).
  // CUDA build: returns the device buffer (allocated by Backend at ctor).
  real_t*       device_ptr();
  const real_t* device_ptr() const;

  void to_device();   // host -> device. CPU build: no-op.
  void to_host();     // device -> host. CPU build: no-op.

  // Halo exchange. Operates on whichever buffer is currently authoritative:
  //   CPU build  : host
  //   CUDA build : device (assumes CUDA-aware MPI; see cuda_aware_mpi.hpp)
  void exchange_halo();

private:
  const parallel::mpi::Subdomain& sub_;
  parallel::Backend&              backend_;
  std::string                     name_;
  std::size_t                     n_elements_ = 0;

  std::vector<real_t> host_;
  real_t*             device_ = nullptr;   // CUDA build owns this; CPU build leaves it null
};

} // namespace mpmstd::field
