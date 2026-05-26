#include "equation/scalar/kernels/kernels.hpp"
#include "common/macros.hpp"

namespace mpmstd::equation::scalar::kernels {

namespace {

// Linear index for the (i, j, k) cell in a halo'd 3-D array of shape
//   (n1_tot, n2_tot, n3_tot)  with strides (n2_tot*n3_tot, n3_tot, 1).
MPMSTD_HD int idx3(int i, int j, int k, int /*n1*/, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

} // anonymous namespace

// =============================================================================
//  (1) Explicit Laplacian
// =============================================================================
void laplacian_explicit_rhs(real_t*       rhs,
                              const real_t* phi,
                              const real_t* dx1, const real_t* dmx1,
                              const real_t* dx2, const real_t* dmx2,
                              const real_t* dx3, const real_t* dmx3,
                              int n1_tot, int n2_tot, int n3_tot,
                              real_t alpha_dt) {
  // Interior cells only: indices [kHaloWidth, n_tot - kHaloWidth).
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int c   = idx3(i,   j,   k,   n1_tot, n2_tot, n3_tot);
        const int xm  = idx3(i-1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int xp  = idx3(i+1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int ym  = idx3(i,   j-1, k,   n1_tot, n2_tot, n3_tot);
        const int yp  = idx3(i,   j+1, k,   n1_tot, n2_tot, n3_tot);
        const int zm  = idx3(i,   j,   k-1, n1_tot, n2_tot, n3_tot);
        const int zp  = idx3(i,   j,   k+1, n1_tot, n2_tot, n3_tot);

        // L_x phi_i = (1/dx_i) [ (phi_{i+1}-phi_i)/dmx_{i+1}
        //                      - (phi_i - phi_{i-1})/dmx_{i}    ]
        const real_t Lx = ((phi[xp] - phi[c]) / dmx1[i+1]
                         - (phi[c] - phi[xm]) / dmx1[i  ]) / dx1[i];
        const real_t Ly = ((phi[yp] - phi[c]) / dmx2[j+1]
                         - (phi[c] - phi[ym]) / dmx2[j  ]) / dx2[j];
        const real_t Lz = ((phi[zp] - phi[c]) / dmx3[k+1]
                         - (phi[c] - phi[zm]) / dmx3[k  ]) / dx3[k];

        rhs[c] = alpha_dt * (Lx + Ly + Lz);
      }
}


// =============================================================================
//  (1b) Convection term — conservative flux form for cell-centered phi
//        with face-centered velocity (U on +x face, V on +y face, W on +z).
// =============================================================================
void add_convection_rhs(real_t*       rhs,
                          const real_t* phi,
                          const real_t* U, const real_t* V, const real_t* W,
                          const real_t* dx1, const real_t* dx2, const real_t* dx3,
                          int n1_tot, int n2_tot, int n3_tot,
                          real_t neg_dt) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int c   = idx3(i,   j,   k,   n1_tot, n2_tot, n3_tot);
        const int xm  = idx3(i-1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int xp  = idx3(i+1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int ym  = idx3(i,   j-1, k,   n1_tot, n2_tot, n3_tot);
        const int yp  = idx3(i,   j+1, k,   n1_tot, n2_tot, n3_tot);
        const int zm  = idx3(i,   j,   k-1, n1_tot, n2_tot, n3_tot);
        const int zp  = idx3(i,   j,   k+1, n1_tot, n2_tot, n3_tot);

        // Conservative ∂(uφ)/∂x at cell i: face fluxes are computed from
        // the upstream/downstream cell values via linear interpolation.
        const real_t flux_xp = U[xp] * 0.5 * (phi[c]  + phi[xp]);
        const real_t flux_xm = U[c ] * 0.5 * (phi[xm] + phi[c ]);
        const real_t flux_yp = V[yp] * 0.5 * (phi[c]  + phi[yp]);
        const real_t flux_ym = V[c ] * 0.5 * (phi[ym] + phi[c ]);
        const real_t flux_zp = W[zp] * 0.5 * (phi[c]  + phi[zp]);
        const real_t flux_zm = W[c ] * 0.5 * (phi[zm] + phi[c ]);

        const real_t div_uphi = (flux_xp - flux_xm) / dx1[i]
                              + (flux_yp - flux_ym) / dx2[j]
                              + (flux_zp - flux_zm) / dx3[k];

        rhs[c] += neg_dt * div_uphi;
      }
}


// =============================================================================
//  (2) ADI tridiagonal bands + RHS pack
//
//      The discrete operator on axis d at cell i is
//          L_d phi_i = (1/dx_i)*(  (phi_{i+1}-phi_i)/dmx_{i+1}
//                                 - (phi_i - phi_{i-1})/dmx_{i} )
//
//      so the implicit matrix (I - 0.5*dt*alpha*L_d) has bands
//          A_i = -0.5*dt*alpha / (dx_i * dmx_i)        (sub-diagonal)
//          C_i = -0.5*dt*alpha / (dx_i * dmx_{i+1})    (super-diagonal)
//          B_i =  1 - A_i - C_i                        (diagonal)
//
//      These are *cell-centered* coefficients, evaluated at the row's
//      cell index along the chosen axis.  The same coefficients apply to
//      every column of the [n_row × n_sys] band buffer (because the metric
//      is axis-1-D), but we still fill them in element-wise for clarity.
// =============================================================================
void build_adi_bands(Direction      d,
                     real_t*        A, real_t* B, real_t* C, real_t* D,
                     const real_t*  src,
                     const real_t*  dx_along, const real_t* dmx_along,
                     int            n1_tot, int n2_tot, int n3_tot,
                     int            n1_int, int n2_int, int n3_int,
                     real_t         alpha, real_t dt) {
  const real_t half_dt_alpha = static_cast<real_t>(0.5) * dt * alpha;

  // (sys, row) <- (i_int, j_int, k_int) mapping switches by axis.
  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii) {
      const int row = ii;
      const int i   = ii + kHaloWidth;       // halo offset
      const real_t a = -half_dt_alpha / (dx_along[i] * dmx_along[i  ]);
      const real_t c = -half_dt_alpha / (dx_along[i] * dmx_along[i+1]);
      const real_t b = static_cast<real_t>(1.0) - a - c;
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = jj * n3_int + kk;
          const int p   = row * n_sys + sys;
          A[p] = a;  B[p] = b;  C[p] = c;
          D[p] = src[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                            n1_tot, n2_tot, n3_tot)];
        }
    }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int row = jj;
      const int j   = jj + kHaloWidth;
      const real_t a = -half_dt_alpha / (dx_along[j] * dmx_along[j  ]);
      const real_t c = -half_dt_alpha / (dx_along[j] * dmx_along[j+1]);
      const real_t b = static_cast<real_t>(1.0) - a - c;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = ii * n3_int + kk;
          const int p   = row * n_sys + sys;
          A[p] = a;  B[p] = b;  C[p] = c;
          D[p] = src[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                            n1_tot, n2_tot, n3_tot)];
        }
    }
  } else { // Direction::Z
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk) {
      const int row = kk;
      const int k   = kk + kHaloWidth;
      const real_t a = -half_dt_alpha / (dx_along[k] * dmx_along[k  ]);
      const real_t c = -half_dt_alpha / (dx_along[k] * dmx_along[k+1]);
      const real_t b = static_cast<real_t>(1.0) - a - c;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int sys = ii * n2_int + jj;
          const int p   = row * n_sys + sys;
          A[p] = a;  B[p] = b;  C[p] = c;
          D[p] = src[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                            n1_tot, n2_tot, n3_tot)];
        }
    }
  }
}


// =============================================================================
//  (3) Scatter TDMA solution back into the 3-D buffer (interior only).
// =============================================================================
void scatter_from_tdma(Direction      d,
                        real_t*        dst,
                        const real_t*  D_solution,
                        int            n1_tot, int n2_tot, int n3_tot,
                        int            n1_int, int n2_int, int n3_int) {
  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii)
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = jj * n3_int + kk;
          const int p   = ii * n_sys + sys;
          dst[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                    n1_tot, n2_tot, n3_tot)] = D_solution[p];
        }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj)
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = ii * n3_int + kk;
          const int p   = jj * n_sys + sys;
          dst[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                    n1_tot, n2_tot, n3_tot)] = D_solution[p];
        }
  } else { // Direction::Z
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk)
      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int sys = ii * n2_int + jj;
          const int p   = kk * n_sys + sys;
          dst[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                    n1_tot, n2_tot, n3_tot)] = D_solution[p];
        }
  }
}


// =============================================================================
//  (4) T += delta on interior cells.
// =============================================================================
void add_increment(real_t*       T,
                    const real_t* delta,
                    int           n1_tot, int n2_tot, int n3_tot) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int p = idx3(i, j, k, n1_tot, n2_tot, n3_tot);
        T[p] += delta[p];
      }
}

} // namespace mpmstd::equation::scalar::kernels
