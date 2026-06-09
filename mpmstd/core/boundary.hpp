#pragma once

// Host-single boundary description, input-driven & BC-agnostic (rev.2 §9c): the
// solver derives sweep order / cyclic-vs-solve / ghost fill / matrix-row mod /
// Poisson transform from this. Re-exported from the existing tree.
#include "boundary/problem.hpp"
#include "boundary/bc_kind.hpp"

namespace mpmstd::core {
using Boundary          = boundary::Problem;   // §5 legacy name
using BoundaryCondition = boundary::Problem;   // rev.2 redesign name (preferred)
using BcKind            = boundary::BcKind;
} // namespace mpmstd::core
