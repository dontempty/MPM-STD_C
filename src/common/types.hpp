#pragma once

#include <cstddef>
#include <cstdint>

namespace mpmstd {

#ifdef MPMSTD_SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

using int_t   = int;
using index_t = std::ptrdiff_t;
using size_t  = std::size_t;

} // namespace mpmstd
