#pragma once

#include "core/bands.hpp"

namespace mpmstd::solve {

// =============================================================================
// solve/  — common, stable, IMPLICIT-only linear solvers (rev.2 §2). Every
// equation's solve_* calls into here; the layer never changes per problem.
// -----------------------------------------------------------------------------
// solve_banded_cpu(): solve the batched banded system carried by a core::Bands
// in place (rhs <- solution). For the spike this is a self-contained Thomas
// sweep that VALIDATES the assemble -> solve data path + signature. In P1 the
// body is replaced by the PaScaL_TDMA / PTDMA backend (with inter-rank
// communication, cyclic vs non-cyclic chosen from the BC) behind this SAME
// signature — the spike exists to freeze that signature.
// =============================================================================
void solve_banded_cpu(core::Bands& b);

} // namespace mpmstd::solve
