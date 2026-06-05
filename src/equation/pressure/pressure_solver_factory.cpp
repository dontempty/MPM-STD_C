#include "equation/pressure/pressure_solver_factory.hpp"
#include "equation/pressure/dct_pressure_solver.hpp"
#include "equation/pressure/pressure_equation.hpp"

#include <stdexcept>

namespace mpmstd::equation::pressure {

std::unique_ptr<PressureSolverBase>
make_pressure_solver(const grid::Grid&                  grid,
                     const parallel::mpi::Subdomain&    sub,
                     field::FieldRegistry&              fields,
                     const boundary::Problem&           problem,
                     linear_solver::tdma::TdmaRegistry& tdma,
                     boundary::BoundaryApplier&         bc) {
  const bool periodic_x = problem.topology.is_periodic(Direction::X);
  const bool periodic_y = problem.topology.is_periodic(Direction::Y);

  if (periodic_x && periodic_y) {
    return std::make_unique<PressureSolver>(grid, sub, fields, problem, tdma, bc);
  } else if (!periodic_x && !periodic_y) {
    return std::make_unique<DctPressureSolver>(grid, sub, fields, problem, tdma, bc);
  } else {
    throw std::runtime_error(
      "make_pressure_solver: mixed periodic/Neumann in X and Y is not yet supported. "
      "Both must be periodic (FFT) or both non-periodic (DCT).");
  }
}

} // namespace mpmstd::equation::pressure
