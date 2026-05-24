#pragma once

#include "parallel/backend/backend.hpp"

namespace mpmstd::parallel {

class CpuBackend : public Backend {
public:
  CpuBackend();
  ~CpuBackend() override = default;

  void* alloc(std::size_t bytes) override;
  void  free (void* ptr) override;
  void  synchronize() override;
  const std::string& name() const override { return name_; }

private:
  std::string name_;
};

} // namespace mpmstd::parallel
