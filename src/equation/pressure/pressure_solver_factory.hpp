#pragma once

#include "boundary/boundary_applier.hpp"
#include "boundary/problem.hpp"
#include "equation/pressure/pressure_solver_base.hpp"
#include "field/field_registry.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <memory>

namespace mpmstd::equation::pressure {

// Factory: selects the appropriate pressure solver from the Problem topology.
//
//   Periodic  X and Y  →  PressureSolver   (2D FFT + z-TDMA)
//   Neumann   X or Y   →  DctPressureSolver (DCT + z-TDMA, M5+)
//
// The caller owns the returned object.
std::unique_ptr<PressureSolverBase>
make_pressure_solver(const grid::Grid&                  grid,
                     const parallel::mpi::Subdomain&    sub,
                     field::FieldRegistry&              fields,
                     const boundary::Problem&           problem,
                     linear_solver::tdma::TdmaRegistry& tdma,
                     boundary::BoundaryApplier&         bc);

} // namespace mpmstd::equation::pressure
