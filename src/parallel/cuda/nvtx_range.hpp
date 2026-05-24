#pragma once

// RAII range marker for NVIDIA Nsight profiling.
// CUDA build : pushes/pops an NVTX range.
// CPU build  : no-op (zero overhead).

#ifdef MPMSTD_BACKEND_CUDA
  #include <nvtx3/nvToolsExt.h>
#endif

namespace mpmstd::parallel::cuda_helpers {

class NvtxRange {
public:
  explicit NvtxRange(const char* name) {
#ifdef MPMSTD_BACKEND_CUDA
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~NvtxRange() {
#ifdef MPMSTD_BACKEND_CUDA
    nvtxRangePop();
#endif
  }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

} // namespace mpmstd::parallel::cuda_helpers

// Convenience macro: scope-block range. Pass the literal name.
#define MPMSTD_NVTX_RANGE(name) \
  ::mpmstd::parallel::cuda_helpers::NvtxRange _mpmstd_nvtx_range_##__LINE__(name)
