#pragma once

#include "common/types.hpp"
#include "common/direction.hpp"

namespace mpmstd::equation::momentum::kernels {

// ── MPM-STD BW-ADI explicit RHS ──────────────────────────────────────────────
// rhs[c] = dt * nu * (Lx + Ly + Lz)(q)[c]
//        - conv_factor * dt * div(u*q)[c]
//
// Standard Laplacian only (no own-direction doubling).
// Backward-Euler for own direction is handled by visc_factor=1.0 in build_adi_bands.
// Convection uses 0.25 coefficient (semi-implicit: rest goes into ADI bands).
// Interior cells only; halos must be filled before this call.
// conv_factor controls the convection contribution: 0.25 normally, 0 to disable.
void compute_mpmstd_rhs(real_t*       rhs,
                         const real_t* q,
                         const real_t* U, const real_t* V, const real_t* W,
                         const real_t* dx1, const real_t* dmx1,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1_tot, int n2_tot, int n3_tot,
                         real_t viscosity, real_t dt,
                         Direction own_dir,
                         bool z_staggered = false,
                         real_t conv_factor = real_t{0.25});

// ── Explicit source accumulation ─────────────────────────────────────────────
// rhs[c] += dt * src[c]  (interior only)
void add_source_rhs(real_t*       rhs,
                     const real_t* src,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt);

// ── Uniform body force ───────────────────────────────────────────────────────
// rhs[c] += val  (interior only)  — used for mean pressure gradient in channel flow
void add_constant_rhs(real_t* rhs,
                       int n1_tot, int n2_tot, int n3_tot,
                       real_t val);

// ── ADI band assembly (MPM-STD BW-ADI) ──────────────────────────────────────
// Build (A, B, C) bands and pack src into D for the tridiagonal system
//   (I − visc_factor·dt·ν·L_d − 0.25·dt·conv_d) δ = src
// along direction d.  Layout: D[row * n_sys + sys].
//
// visc_factor = 1.0 for the "own" direction (backward Euler)
// visc_factor = 0.5 for cross directions   (Crank-Nicolson)
//
// q      : current field values (for convective terms in bands)
// q_adv  : advecting velocity in direction d (U for X, V for Y, W for Z)
void build_adi_bands(Direction      d,
                      real_t*        A, real_t* B, real_t* C, real_t* D,
                      const real_t*  src,
                      const real_t*  q,
                      const real_t*  q_adv,
                      const real_t*  dx_along, const real_t* dmx_along,
                      int            n1_tot, int n2_tot, int n3_tot,
                      int            n1_int,  int n2_int,  int n3_int,
                      real_t         viscosity, real_t dt,
                      real_t         visc_factor = real_t{0.5},
                      bool           z_staggered = false);

// ── Scatter TDMA solution back to 3-D buffer ─────────────────────────────────
void scatter_from_tdma(Direction      d,
                        real_t*        dst,
                        const real_t*  D_solution,
                        int n1_tot, int n2_tot, int n3_tot,
                        int n1_int,  int n2_int,  int n3_int);

// ── q += delta (interior cells) ──────────────────────────────────────────────
void add_increment(real_t*       q,
                    const real_t* delta,
                    int n1_tot, int n2_tot, int n3_tot);

// ── MPM-STD block-coupling correction (constant viscosity, convection part) ──
//
// The fully-coupled MPM-STD momentum solve treats the off-diagonal velocity
// blocks (how δV, δW feed back into the U-equation, and δW into the V-equation)
// with a block lower-triangular Gauss-Seidel step applied AFTER the three
// diagonal ADI predictors and BEFORE the velocity update:
//
//   dV ← dV − dt·0.25·( dW·∂V/∂z )                      (blockLdV: M23·dW)
//   dU ← dU − dt·0.25·( dV·∂U/∂y + dW·∂U/∂z )           (blockLdU: M12·dV + M13·dW)
//
// This cross-velocity convection of the increments is the streak↔vortex coupling
// that sustains near-wall turbulence; omitting it lets the channel relaminarize.
// The variable-viscosity cross-stress terms (∂/∂y(μ ∂δV/∂x), …) are dropped here
// because for constant viscosity they are a continuity correction that the
// pressure projection already enforces.
//
// dU, dV, dW are the increment buffers (must have valid halos for the stencil).
// U, V are the current-step velocity fields (u^n).  Interior cells only.

// dV −= dt·0.25·(dW·∂V/∂z) interpolated to V (y-face) locations.
void block_couple_dV(real_t*       dV,
                     const real_t* dW,
                     const real_t* V,
                     const real_t* dx2, const real_t* dmx2,
                     const real_t* dx3, const real_t* dmx3,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt);

// dU −= dt·0.25·(dV·∂U/∂y + dW·∂U/∂z) interpolated to U (x-face) locations.
void block_couple_dU(real_t*       dU,
                     const real_t* dV,
                     const real_t* dW,
                     const real_t* U,
                     const real_t* dx1, const real_t* dmx1,
                     const real_t* dmx2,
                     const real_t* dmx3,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt);

} // namespace mpmstd::equation::momentum::kernels
