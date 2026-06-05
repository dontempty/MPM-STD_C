#pragma once

#include "common/types.hpp"
#include "field/scalar_field.hpp"

namespace mpmstd::physics {

// ── Boussinesq approximation parameters ─────────────────────────────────────
//
// Dimensionless form used in RBC:
//
//   ∂T/∂t + u·∇T = (1 / Ra·Pr)^{1/2} * (1/Pr)  ∇²T          [wrong]
//
// Actually the standard dimensionless NOB equations scaled by the free-fall
// velocity and the domain height H give:
//
//   ∂u/∂t + u·∇u = -(1/ρ)∇p + (Pr/Ra)^{1/2} ∇²u + T ẑ
//   ∂T/∂t + u·∇T =  (1/Ra·Pr)^{1/2}            ∇²T
//
// For Boussinesq (constant property) the buoyancy body force on the
// z-velocity is simply:
//
//   f_z = T                (if Ra is folded into the time-scale)
//
// The caller sets the viscosity in MomentumTraits to (Pr/Ra)^{1/2}
// and computes the source field via compute_z_buoyancy() before each step.

struct BoussinesqParams {
  real_t Ra;    // Rayleigh number
  real_t Pr;    // Prandtl number

  // Derived quantities.
  real_t viscosity()   const;  // (Pr/Ra)^{1/2}
  real_t diffusivity() const;  // 1 / (Ra·Pr)^{1/2}
};

// Fill `src` with the z-buoyancy contribution for the W-momentum equation.
// Boussinesq: f_z = T  (no scaling — already folded into the velocity scale).
// src and T must have the same subdomain shape.
void compute_z_buoyancy(field::ScalarField&       src,
                         const field::ScalarField& T);

} // namespace mpmstd::physics
