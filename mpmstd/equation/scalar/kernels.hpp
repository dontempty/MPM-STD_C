#pragma once

#include "common/types.hpp"
#include "common/direction.hpp"

namespace mpmstd::equation::scalar::kernels {

// =============================================================================
//  Layer 1: physics kernels  (compiled per backend; same signatures)
//
//  All functions operate on **flat row-major arrays**.  The 3-D field layout
//  is the standard mpmstd one: index (i, j, k) → (i*n2_tot + j)*n3_tot + k,
//  with halo width 1 on every axis (so interior indices run 1..n_int along
//  each axis).
//
//  For TDMA, the bands (A, B, C, D) are laid out in PaScaL_TDMA's expected
//  [n_row × n_sys] row-major form.  The mapping between (i, j, k) and
//  (sys, row) is axis-specific and handled by the pack/unpack helpers below.
// =============================================================================


// -----------------------------------------------------------------------------
//  (1) Explicit full Laplacian: rhs += dt * alpha * (Lx + Ly + Lz) phi
//
//      Lx phi at cell i  =  (1/dx_i) [ (phi_{i+1} - phi_i)/dmx_{i+1}
//                                     - (phi_i - phi_{i-1})/dmx_{i} ]
//
//  `rhs` is written on the interior cells only (interior shape exactly
//  matches `phi`'s interior — the entire n_total^3 array, but only the
//  central (n_int)^3 cells are touched).
// -----------------------------------------------------------------------------
void laplacian_explicit_rhs(real_t*       rhs,
                              const real_t* phi,
                              const real_t* dx1, const real_t* dmx1,
                              const real_t* dx2, const real_t* dmx2,
                              const real_t* dx3, const real_t* dmx3,
                              int n1_tot, int n2_tot, int n3_tot,
                              real_t alpha_dt);


// -----------------------------------------------------------------------------
//  (1b) Convection term:  rhs += -dt * u·∇phi
//
//  Conservative flux form for cell-centered phi with face-centered velocity
//  (PaScaL_TCS convention — U is on the +x face of cell i, etc.):
//
//      ∂(uφ)/∂x at cell i =  ( U[i+1] * 0.5 (φ[i] + φ[i+1])
//                            - U[i  ] * 0.5 (φ[i-1] + φ[i]) ) / dx_i
//
//  ∇·(uφ) is then  ∂(uφ)/∂x + ∂(vφ)/∂y + ∂(wφ)/∂z.
//
//  `neg_dt` is typically -dt (the caller passes the explicit-step factor).
//  Halo cells of φ and the velocity fields must be filled before this call.
// -----------------------------------------------------------------------------
void add_convection_rhs(real_t*       rhs,
                          const real_t* phi,
                          const real_t* U, const real_t* V, const real_t* W,
                          const real_t* dx1, const real_t* dx2, const real_t* dx3,
                          int n1_tot, int n2_tot, int n3_tot,
                          real_t neg_dt);


// -----------------------------------------------------------------------------
//  (2) Build ADI tridiagonal bands for ONE axis.
//
//  For each axis d, the bands (A, B, C) of the operator (I - 0.5*dt*alpha*L_d)
//  are built and packed into the [n_row × n_sys] layout PaScaL_TDMA wants:
//
//      idx = row * n_sys + sys
//
//  Mapping (interior 0-based) -> (sys, row):
//      d = X : row = i_int,  sys = j_int * n3_int + k_int
//      d = Y : row = j_int,  sys = i_int * n3_int + k_int
//      d = Z : row = k_int,  sys = i_int * n2_int + j_int
//
//  `src` is the RHS as a flat n_total^3 array (interior n_int^3 cells used);
//  it is copied into D with the same axis-specific packing as the bands.
//
//  For the CYCLIC (periodic-axis) TDMA, the matrix is the same as the
//  non-cyclic interior bands — PaScaL_TDMA handles the wrap-around itself.
//  For the NON-CYCLIC case, BoundaryApplier::modify_tdma_row will (M2 Phase 2)
//  amend the boundary rows; for now Phase 1 supports periodic axes only.
// -----------------------------------------------------------------------------
void build_adi_bands(Direction      d,
                     real_t*        A, real_t* B, real_t* C, real_t* D,
                     const real_t*  src,
                     const real_t*  dx_along, const real_t* dmx_along,
                     int            n1_tot, int n2_tot, int n3_tot,
                     int            n1_int, int n2_int, int n3_int,
                     real_t         alpha, real_t dt);


// -----------------------------------------------------------------------------
//  (3) Scatter the TDMA solution back into a flat n_total^3 array.
//
//      Reverses the (sys, row) -> (i, j, k) mapping used by build_adi_bands.
//      Only interior cells are written.
// -----------------------------------------------------------------------------
void scatter_from_tdma(Direction d,
                        real_t*        dst,
                        const real_t*  D_solution,
                        int            n1_tot, int n2_tot, int n3_tot,
                        int            n1_int, int n2_int, int n3_int);


// -----------------------------------------------------------------------------
//  (4) Add `delta` (interior cells of an n_total^3 buffer) to `T`.
// -----------------------------------------------------------------------------
void add_increment(real_t*       T,
                    const real_t* delta,
                    int           n1_tot, int n2_tot, int n3_tot);

} // namespace mpmstd::equation::scalar::kernels
