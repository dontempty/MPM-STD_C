#pragma once

#include "parallel/backend/backend.hpp"

namespace mpmstd::parallel {

// Real CUDA implementation lands in M5'.
// In CPU build this class is still declared so callers' types resolve, but it
// is never instantiated.

class CudaBackend : public Backend {
public:
  CudaBackend();
  ~CudaBackend() override;

  void* alloc(std::size_t bytes) override;
  void  free (void* ptr) override;
  void  synchronize() override;
  const std::string& name() const override { return name_; }

private:
  std::string name_;
};

} // namespace mpmstd::parallel
