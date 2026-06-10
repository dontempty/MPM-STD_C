#pragma once

#include "common/types.hpp"        // real_t
#include "common/direction.hpp"    // Direction, to_int
#include "parallel/mpi/subdomain.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace mpmstd::core {

// =============================================================================
// GpuField  (rev.2 C1=(b): device-resident twin of CpuField)
// -----------------------------------------------------------------------------
// Same logical layout as CpuField; the buffer lives in device memory
// (cudaMalloc). This HEADER is backend-agnostic (no <cuda_runtime.h>) so it can
// be included in any translation unit for declarations; the member definitions
// (ctor/dtor/copies) are in core/gpu/field_gpu.cu and are compiled ONLY in the
// GPU build (the gpu/ folder is skipped by `make cpu`, so the CPU-only build has
// zero CUDA dependency — rev.2 §6).
//
// to_device/to_host are for IO + regression comparison against CpuField, NOT
// for a hot-loop host<->device shuffle (rev.2 keeps one backend per app).
// =============================================================================
class GpuField {
public:
  GpuField(const parallel::mpi::Subdomain& sub, std::string name);
  ~GpuField();

  GpuField(const GpuField&)            = delete;
  GpuField& operator=(const GpuField&) = delete;

  const std::string& name() const { return name_; }

  std::array<int, 3> n_total()       const { return n_total_; }
  std::array<int, 3> n_interior()    const { return n_interior_; }
  std::array<int, 3> global_offset() const { return offset_; }
  int n_total(Direction d)       const { return n_total_[to_int(d)]; }
  int n_interior(Direction d)    const { return n_interior_[to_int(d)]; }
  int global_offset(Direction d) const { return offset_[to_int(d)]; }
  std::size_t n_elements() const { return n_elements_; }

  int linear_index(int i, int j, int k) const {
    return (i * n_total_[1] + j) * n_total_[2] + k;
  }

  // device buffer
  real_t*       data()       { return device_; }
  const real_t* data() const { return device_; }

  // host <-> device (IO / regression). host_* must have n_elements() entries.
  void to_device(const real_t* host_src);
  void to_host  (real_t* host_dst) const;

private:
  std::string        name_;
  std::array<int, 3> n_total_{};
  std::array<int, 3> n_interior_{};
  std::array<int, 3> offset_{};
  std::size_t        n_elements_ = 0;
  real_t*            device_     = nullptr;
};

} // namespace mpmstd::core
