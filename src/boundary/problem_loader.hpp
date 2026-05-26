#pragma once

#include "boundary/problem.hpp"
#include "config/config.hpp"

namespace mpmstd::boundary {

// Build a Problem from a Config file.
//
// Config schema
// -------------
//   [topology]
//   x = "periodic"     ; one of: "periodic", "dirichlet", "neumann", "wall"
//   y = "periodic"
//   z = "wall"
//
//   ; [bc.<axis>.<side>] entries override the *value* of the per-field BC
//   ; on a single global face.  Missing entries default to 0.  Whether a
//   ; field can be overridden depends on the axis's topology kind (see below).
//   [bc.z.minus]
//   T = -0.5
//
//   [bc.z.plus]
//   T = 0.5
//
// Semantics of topology kinds
// ---------------------------
//   periodic    Periodic axis. All field faces on this axis are Periodic.
//               [bc.<axis>.*] is ignored.
//
//   dirichlet   Non-periodic axis. Every field (U, V, W, P, T) is Dirichlet.
//               [bc.<axis>.<side>] may override any of them; default 0.
//
//   neumann     Non-periodic axis. Every field (U, V, W, P, T) is Neumann.
//               Values are always 0 (zero-gradient); [bc.*] is ignored.
//
//   wall        Non-periodic axis. Physical wall:
//                  U, V, W, T  -> Dirichlet (default 0; user-overridable)
//                  P           -> Neumann   (always 0; never overridable)
//               This is the canonical kind for RBC / channel walls.
//
// Future extension
// ----------------
//   FaceBc carries a std::function<real_t(x,y,z,t)>, so the loader could be
//   extended to accept expression strings (e.g. T = "0.5*cos(2*pi*x/Lx)")
//   without changing any callers.  Not implemented yet.

Problem load_problem_from_config(const config::Config& cfg);

} // namespace mpmstd::boundary
