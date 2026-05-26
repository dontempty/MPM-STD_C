#pragma once

#include "common/types.hpp"

#include <string>

namespace mpmstd::equation::scalar {

// ScalarTraits = small POD describing one passive-scalar / temperature variable.
//
//   name              : the FieldRegistry key (e.g. "T", "Y_O2", ...).
//   diffusivity       : the molecular diffusion coefficient α appearing in
//                       ∂φ/∂t + u·∇φ = α ∇²φ + source.
//   with_convection   : if true, ScalarEquation will read the velocity
//                       components ("U", "V", "W") from the same
//                       FieldRegistry and include −u·∇φ in the explicit RHS.
//                       If false, ScalarEquation solves pure diffusion.
//
// More members will be added later (source term function, per-scalar
// property policy for variable α(T), custom velocity field names, ...).
// Phase-1/2 used only `name` and `diffusivity`; Phase 3 adds `with_convection`.

struct ScalarTraits {
  std::string name;
  real_t      diffusivity;
  bool        with_convection = false;
};

} // namespace mpmstd::equation::scalar
