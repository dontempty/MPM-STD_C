#pragma once

#include "common/types.hpp"
#include <cstddef>
#include <string>

namespace mpmstd::parallel {

// Backend abstracts memory allocation and stream/queue management.
// Hot-loop kernels are NOT methods on this interface; instead, equation
// modules call into kernel functions whose CPU/CUDA implementations are
// chosen at build time (see equation/<name>/kernels/kernels_cpu.cpp vs
// kernels_cuda.cu).
//
// This keeps Backend small and stable.

class Backend {
public:
  virtual ~Backend() = default;

  // Allocate `bytes` bytes of memory accessible from the compute device.
  // CPU build : aligned host memory.
  // CUDA build: cudaMalloc.
  virtual void* alloc(std::size_t bytes) = 0;
  virtual void  free (void* ptr) = 0;

  // Synchronize the default stream / device.
  virtual void synchronize() = 0;

  // Human-readable identifier for logging.
  virtual const std::string& name() const = 0;
};

} // namespace mpmstd::parallel
