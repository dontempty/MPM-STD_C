#pragma once

// Convenience wrapper around the CUDA_CHECK macro defined in common/macros.hpp.
// This header exists so that callers can write
//     #include "parallel/cuda/error_check.hpp"
// without pulling in the full common/main.hpp.

#include "common/macros.hpp"
