#pragma once

#include "common/macros.hpp"
#include "common/types.hpp"
#include "field/stencil/staggered.hpp"

namespace mpmstd::field::stencil {

// =============================================================================
// Face-staggered harmonic mean of a cell-centered quantity.
//
// PaScaL_TCS interpolation for viscosity at the y- and z-faces around a
// staggered U-velocity location (see module_solve_momentum.f90:317-329).
//
//   muc = 0.5 / dmx1(i) * ( dx1(im)*Mu(i, j, k) + dx1(i)*Mu(im, j, k) )
//   mu3 = 0.5 / dmx2(j) * ( dx2(jm)*muc          + dx2(j) *mua          )
//
// These helpers express the same arithmetic at a single (i,j,k) point.  The
// indexing conventions assume the same row-major storage as `staggered.hpp`.
//
// These are NOT meant to compute the full viscous flux; they are building
// blocks invoked by the momentum RHS assembly (M3).
// =============================================================================

// Linear interpolation of `phi` from the two cell centers (i-1, j, k) and
// (i, j, k) onto the +x face at xf[i]. Weights are the *opposite* cell widths,
// which makes the result exact on non-uniform grids when phi is linear.
MPMSTD_HD real_t face_interp_x(const real_t* MPMSTD_RESTRICT phi,
                                const real_t* MPMSTD_RESTRICT dx1,
                                const real_t* MPMSTD_RESTRICT dmx1,
                                int i, int j, int k,
                                int n1, int n2, int n3) {
  const int idx_c  = linear_index(i,   j, k, n1, n2, n3);
  const int idx_im = linear_index(i-1, j, k, n1, n2, n3);
  return real_t(0.5) / dmx1[i] * (dx1[i-1] * phi[idx_c] + dx1[i] * phi[idx_im]);
}

MPMSTD_HD real_t face_interp_y(const real_t* MPMSTD_RESTRICT phi,
                                const real_t* MPMSTD_RESTRICT dx2,
                                const real_t* MPMSTD_RESTRICT dmx2,
                                int i, int j, int k,
                                int n1, int n2, int n3) {
  const int idx_c  = linear_index(i, j,   k, n1, n2, n3);
  const int idx_jm = linear_index(i, j-1, k, n1, n2, n3);
  return real_t(0.5) / dmx2[j] * (dx2[j-1] * phi[idx_c] + dx2[j] * phi[idx_jm]);
}

MPMSTD_HD real_t face_interp_z(const real_t* MPMSTD_RESTRICT phi,
                                const real_t* MPMSTD_RESTRICT dx3,
                                const real_t* MPMSTD_RESTRICT dmx3,
                                int i, int j, int k,
                                int n1, int n2, int n3) {
  const int idx_c  = linear_index(i, j, k,   n1, n2, n3);
  const int idx_km = linear_index(i, j, k-1, n1, n2, n3);
  return real_t(0.5) / dmx3[k] * (dx3[k-1] * phi[idx_c] + dx3[k] * phi[idx_km]);
}

} // namespace mpmstd::field::stencil
