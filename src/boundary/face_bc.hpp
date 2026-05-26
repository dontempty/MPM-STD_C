#pragma once

#include "boundary/bc_kind.hpp"
#include "common/types.hpp"

#include <functional>

namespace mpmstd::boundary {

// Value-evaluation signature: (x, y, z, t) -> field value at that location.
// For a constant BC we just capture a literal and ignore the args.
using BcValueFn = std::function<real_t(real_t, real_t, real_t, real_t)>;

// FaceBc is a small POD-ish struct: kind + value function.
struct FaceBc {
  BcKind    kind  = BcKind::Periodic;
  BcValueFn value = constant(0.0);

  // ----- helper factories (preferred over manual construction) -----
  static FaceBc periodic();
  static FaceBc dirichlet(real_t v);
  static FaceBc dirichlet(BcValueFn f);
  static FaceBc neumann(real_t v);
  static FaceBc neumann(BcValueFn f);

  // Convenience constant-value lambda.
  static BcValueFn constant(real_t v);
};

} // namespace mpmstd::boundary
