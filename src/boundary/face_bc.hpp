#pragma once

#include "boundary/bc_kind.hpp"
#include "common/types.hpp"

namespace mpmstd::boundary {

// Ghost-cell filling policy for Dirichlet faces.
//
//   ZeroGhost      ghost = v_wall                    (1st-order; wall velocity, pressure)
//   Antisymmetric  ghost = 2*v_wall - phi_interior   (2nd-order; temperature)
//
// Only Dirichlet faces use this; Neumann/Periodic ignore it.
enum class GhostPolicy { ZeroGhost, Antisymmetric };

constexpr const char* ghost_policy_name(GhostPolicy p) {
  switch (p) {
    case GhostPolicy::ZeroGhost:     return "zero";
    case GhostPolicy::Antisymmetric: return "antisymmetric";
  }
  return "?";
}

// FaceBc — POD boundary descriptor for one domain face.
//
//   `value` is the CURRENT boundary value (at the current simulation time).
//
// rev.2 note on TIME-DEPENDENT boundaries: a varying wall value v(t) is NOT
// stored as a callback here. The previous std::function approach (a) made
// nvc++ 23.7 ICE on dynamic init and (b) cannot be evaluated inside a CUDA
// kernel. Instead, time evolution is an OPERATION (rev.2 "data=struct,
// ops=free function"): a free function updates `value` each step from t, and
// apply_ghost reads the current scalar — POD, so it copies to the GPU and
// evaluates on device. A future spatially-varying profile would add a small
// profile enum + parameters here (still POD), evaluated by the ghost filler.
struct FaceBc {
  BcKind      kind         = BcKind::Periodic;
  real_t      value        = real_t{0};
  GhostPolicy ghost_policy = GhostPolicy::ZeroGhost;

  // ----- helper factories (constant value at setup; for a time-dependent BC,
  //       a free function rewrites `value` each step) -----
  static FaceBc periodic();
  static FaceBc dirichlet(real_t v);
  static FaceBc dirichlet_antisymm(real_t v);   // 2nd-order ghost (e.g. temperature)
  static FaceBc neumann(real_t v);
};

} // namespace mpmstd::boundary
