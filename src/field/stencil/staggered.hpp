#pragma once

#include "common/macros.hpp"
#include "common/types.hpp"

namespace mpmstd::field::stencil {

// =============================================================================
// Linear index helper
//
// Convention: row-major, slowest dimension is X (axis 0), then Y, then Z.
//   idx(i, j, k) = (i * n2 + j) * n3 + k
//
// All stencil routines take the (n1, n2, n3) extents as parameters so that
// they can be inlined in both host loops and device kernels.
// =============================================================================

MPMSTD_HD int linear_index(int i, int j, int k, int /*n1*/, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

// =============================================================================
// Cell-to-face gradient
//
//   - face_grad_x(P, ..., i, j, k) returns dP/dx at the face between cells
//     (i-1, j, k) and (i, j, k), located at xf[i].  In MAC-staggered notation
//     this is the location of U(i, j, k).
//   - dmx1[i] is the cell-center-to-cell-center distance (xc[i] - xc[i-1]).
//
// These mirror PaScaL_TCS:
//     dpdx_face_x  : (P(i) - P(i-1)) / dmx1(i)
// =============================================================================

MPMSTD_HD real_t face_grad_x(const real_t* MPMSTD_RESTRICT phi,
                              const real_t* MPMSTD_RESTRICT dmx1,
                              int i, int j, int k,
                              int n1, int n2, int n3) {
  const int idx_c  = linear_index(i,   j, k, n1, n2, n3);
  const int idx_im = linear_index(i-1, j, k, n1, n2, n3);
  return (phi[idx_c] - phi[idx_im]) / dmx1[i];
}

MPMSTD_HD real_t face_grad_y(const real_t* MPMSTD_RESTRICT phi,
                              const real_t* MPMSTD_RESTRICT dmx2,
                              int i, int j, int k,
                              int n1, int n2, int n3) {
  const int idx_c  = linear_index(i, j,   k, n1, n2, n3);
  const int idx_jm = linear_index(i, j-1, k, n1, n2, n3);
  return (phi[idx_c] - phi[idx_jm]) / dmx2[j];
}

MPMSTD_HD real_t face_grad_z(const real_t* MPMSTD_RESTRICT phi,
                              const real_t* MPMSTD_RESTRICT dmx3,
                              int i, int j, int k,
                              int n1, int n2, int n3) {
  const int idx_c  = linear_index(i, j, k,   n1, n2, n3);
  const int idx_km = linear_index(i, j, k-1, n1, n2, n3);
  return (phi[idx_c] - phi[idx_km]) / dmx3[k];
}

// =============================================================================
// Face-to-cell gradient
//
// face_to_cell_grad_x(U, dx1, i, j, k): given U on +x faces (U[i,j,k] lives at
// xf[i]), compute dU/dx at the cell center (i, j, k) using forward difference.
//
// This corresponds to PaScaL_TCS's divergence stencil for U:
//     (U(i+1,j,k) - U(i,j,k)) / dx1(i)
// =============================================================================

MPMSTD_HD real_t cell_div_contribution_x(const real_t* MPMSTD_RESTRICT u,
                                          const real_t* MPMSTD_RESTRICT dx1,
                                          int i, int j, int k,
                                          int n1, int n2, int n3) {
  const int idx_c  = linear_index(i,   j, k, n1, n2, n3);
  const int idx_ip = linear_index(i+1, j, k, n1, n2, n3);
  return (u[idx_ip] - u[idx_c]) / dx1[i];
}

MPMSTD_HD real_t cell_div_contribution_y(const real_t* MPMSTD_RESTRICT v,
                                          const real_t* MPMSTD_RESTRICT dx2,
                                          int i, int j, int k,
                                          int n1, int n2, int n3) {
  const int idx_c  = linear_index(i, j,   k, n1, n2, n3);
  const int idx_jp = linear_index(i, j+1, k, n1, n2, n3);
  return (v[idx_jp] - v[idx_c]) / dx2[j];
}

MPMSTD_HD real_t cell_div_contribution_z(const real_t* MPMSTD_RESTRICT w,
                                          const real_t* MPMSTD_RESTRICT dx3,
                                          int i, int j, int k,
                                          int n1, int n2, int n3) {
  const int idx_c  = linear_index(i, j, k,   n1, n2, n3);
  const int idx_kp = linear_index(i, j, k+1, n1, n2, n3);
  return (w[idx_kp] - w[idx_c]) / dx3[k];
}

// =============================================================================
// Divergence at a cell center
//
//   div(U)_{i,j,k} = (U(i+1) - U(i))/dx1(i)
//                  + (V(j+1) - V(j))/dx2(j)
//                  + (W(k+1) - W(k))/dx3(k)
//
// U, V, W are the three components of a VectorField stored in separate arrays
// of identical shape (n1, n2, n3).
// =============================================================================

MPMSTD_HD real_t divergence_at_cell(const real_t* MPMSTD_RESTRICT u,
                                     const real_t* MPMSTD_RESTRICT v,
                                     const real_t* MPMSTD_RESTRICT w,
                                     const real_t* MPMSTD_RESTRICT dx1,
                                     const real_t* MPMSTD_RESTRICT dx2,
                                     const real_t* MPMSTD_RESTRICT dx3,
                                     int i, int j, int k,
                                     int n1, int n2, int n3) {
  return cell_div_contribution_x(u, dx1, i, j, k, n1, n2, n3)
       + cell_div_contribution_y(v, dx2, i, j, k, n1, n2, n3)
       + cell_div_contribution_z(w, dx3, i, j, k, n1, n2, n3);
}

} // namespace mpmstd::field::stencil
