#pragma once

// Host-single grid (coordinates + metrics, uniform/tanh stretch). Re-exported
// verbatim from the existing tree (rev.2: host metadata not duplicated per
// backend). P1 ports/refines as needed.
#include "grid/grid.hpp"

namespace mpmstd::core {
using Grid       = grid::Grid;
using AxisConfig = grid::AxisConfig;
} // namespace mpmstd::core
