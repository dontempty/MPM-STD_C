#pragma once

#include "boundary/domain_topology.hpp"
#include "boundary/field_boundary.hpp"

namespace mpmstd::boundary {

// Problem aggregates everything a Solver needs to know about boundary
// conditions:
//   * the axis topology (which axes are periodic vs. non-periodic),
//   * the per-field FaceBc descriptors for U, V, W, P, T.
//
// The default constructor produces a **neutral** state:
//   topology  = (Periodic, Periodic, Periodic)
//   every field face = Periodic
//
// All non-trivial problem setups (RBC, channel, cavity, ...) come from
// `load_problem_from_config(...)` (see problem_loader.hpp) — there are NO
// hardcoded physical defaults in this class.
//
// Direct manual overrides are still supported for tests / one-off cases:
//   Problem p;
//   p.topology.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;
//   p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet(2.5);
//   p.validate();

class Problem {
public:
  Problem();

  // Top-level state.
  DomainTopology topology;

  // Per-field BC tables.  Names mirror PaScaL_TCS:
  //   U,V,W : velocity components
  //   P     : pressure
  //   T     : temperature
  FieldBoundary U;
  FieldBoundary V;
  FieldBoundary W;
  FieldBoundary P;
  FieldBoundary T;

  // Reset to the neutral all-Periodic state.  Called by the constructor;
  // can be called again to wipe a Problem back to neutral before re-filling
  // it from a different source.
  void apply_periodic_defaults();

  // Throw if face periodicity disagrees with topology.is_periodic(d).
  void validate() const;
};

} // namespace mpmstd::boundary
