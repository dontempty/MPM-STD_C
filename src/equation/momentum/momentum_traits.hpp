#pragma once

#include "common/types.hpp"

#include <string>

namespace mpmstd::equation::momentum {

// MomentumTraits — describes one velocity-component equation
//     ∂q/∂t + u·∇q = ν ∇²q + source
//
//   name            : FieldRegistry key ("U", "V", or "W").
//   viscosity       : kinematic viscosity ν.
//   with_convection : include −u·∇q in the explicit RHS.  Default true.
//   source_name     : FieldRegistry key of an explicit body-force / buoyancy
//                     field (cell-centred, same shape as q).  Empty → no source.
//                     The source is accumulated as  dt * source_field  on the
//                     explicit RHS before the ADI stages.

struct MomentumTraits {
  std::string name;
  real_t      viscosity;
  bool        with_convection = true;
  std::string source_name;      // FieldRegistry key added as dt*field (e.g. "T" for buoyancy)
  real_t      constant_source = real_t{0};  // uniform body force (e.g. mean pressure gradient)
};

} // namespace mpmstd::equation::momentum
