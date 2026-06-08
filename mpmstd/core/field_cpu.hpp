#pragma once

#include "common/types.hpp"        // real_t
#include "common/direction.hpp"    // Direction, to_int
#include "common/macros.hpp"       // kHaloWidth
#include "parallel/mpi/subdomain.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace mpmstd::core {

// =============================================================================
// CpuField  (rev.2 C1=(b): CPU/GPU are SEPARATE types)
// -----------------------------------------------------------------------------
// Host-resident 3D field. Cell- or face-centered; the staggered MAC convention
// is encoded by the INDEXING RULES used in stencils, not by the storage shape
// (all of U/V/W/P/T share one shape — mirrors PaScaL_TCS / the old ScalarField).
//
// Design (rev.2): data structures are classes holding DATA + simple accessors;
// all algorithms are FREE FUNCTIONS taking the data explicitly. So CpuField is
// deliberately "dumb": it captures its shape (n_total / n_interior / global
// offset) BY VALUE at construction from the Subdomain and owns its buffer.
// Operations that need MPI communication (halo exchange) take the Subdomain
// explicitly — see exchange_halo_cpu() in core/halo.hpp. This decouples the
// field from a persistent topology reference and matches the "main reads like a
// recipe" goal.
//
// GpuField (field_gpu.hpp) is the device-resident twin with the same logical
// layout; there is NO shared base class and NO virtual Backend (both dropped
// per rev.2). Callers pick CpuField + *_cpu or GpuField + *_gpu and stay on it.
// =============================================================================
class CpuField {
public:
  CpuField(const parallel::mpi::Subdomain& sub, std::string name)
    : name_(std::move(name)),
      n_total_(sub.n_total()),
      n_interior_(sub.n_interior()),
      offset_(sub.global_offset()),
      data_(sub.n_elements(), real_t{0}) {}

  // ----- metadata (captured by value; self-describing) -----
  const std::string& name() const { return name_; }

  std::array<int, 3> n_total()       const { return n_total_; }
  std::array<int, 3> n_interior()    const { return n_interior_; }
  std::array<int, 3> global_offset() const { return offset_; }

  int n_total(Direction d)       const { return n_total_[to_int(d)]; }
  int n_interior(Direction d)    const { return n_interior_[to_int(d)]; }
  int global_offset(Direction d) const { return offset_[to_int(d)]; }

  std::size_t n_elements() const { return data_.size(); }

  // Row-major flat index (i,j,k 0-based, halo positions included).
  // strides: k -> 1, j -> n_total[2], i -> n_total[1]*n_total[2].
  int linear_index(int i, int j, int k) const {
    return (i * n_total_[1] + j) * n_total_[2] + k;
  }

  // ----- data access -----
  real_t*       data()       { return data_.data(); }
  const real_t* data() const { return data_.data(); }

  real_t&       at(int i, int j, int k)       { return data_[linear_index(i, j, k)]; }
  const real_t& at(int i, int j, int k) const { return data_[linear_index(i, j, k)]; }

  void fill(real_t value) { std::fill(data_.begin(), data_.end(), value); }

private:
  std::string         name_;
  std::array<int, 3>  n_total_{};
  std::array<int, 3>  n_interior_{};
  std::array<int, 3>  offset_{};
  std::vector<real_t> data_;
};

} // namespace mpmstd::core
