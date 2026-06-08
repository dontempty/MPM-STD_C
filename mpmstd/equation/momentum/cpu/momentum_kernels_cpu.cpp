#include "equation/momentum/kernels.hpp"   // copied into mpmstd/ (rev.2 P1, self-contained)
#include "common/macros.hpp"

namespace mpmstd::equation::momentum::kernels {

namespace {

MPMSTD_HD int idx3(int i, int j, int k, int /*n1*/, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

} // anonymous namespace


// =============================================================================
//  MPM-STD BW-ADI explicit RHS
//
//  rhs[c] = dt * nu * (Lx + Ly + Lz)(q)[c]
//           - conv_factor * dt * div(u*q)[c]
//
//  Standard Laplacian (all three directions once).
//  Backward-Euler treatment for the own direction is handled by visc_factor=1.0
//  in build_adi_bands; do NOT double the own-direction Laplacian here.
//  Convection coefficient 0.25 → remaining 0.75 is absorbed into ADI bands.
// =============================================================================
void compute_mpmstd_rhs(real_t*       rhs,
                         const real_t* q,
                         const real_t* U, const real_t* V, const real_t* W,
                         const real_t* dx1, const real_t* dmx1,
                         const real_t* dx2, const real_t* dmx2,
                         const real_t* dx3, const real_t* dmx3,
                         int n1_tot, int n2_tot, int n3_tot,
                         real_t viscosity, real_t dt,
                         Direction /*own_dir*/,
                         bool z_staggered,
                         real_t conv_factor) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  const real_t nu_dt = viscosity * dt;
  // conv_factor * (-dt): negative because we subtract conv_factor*div(u*q).
  const real_t conv_coeff = -conv_factor * dt;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int c  = idx3(i,   j,   k,   n1_tot, n2_tot, n3_tot);
        const int xm = idx3(i-1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int xp = idx3(i+1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int ym = idx3(i,   j-1, k,   n1_tot, n2_tot, n3_tot);
        const int yp = idx3(i,   j+1, k,   n1_tot, n2_tot, n3_tot);
        const int zm = idx3(i,   j,   k-1, n1_tot, n2_tot, n3_tot);
        const int zp = idx3(i,   j,   k+1, n1_tot, n2_tot, n3_tot);

        // --- Viscous Laplacian (all three directions) ---
        const real_t Lx = ((q[xp] - q[c]) / dmx1[i+1]
                         - (q[c] - q[xm]) / dmx1[i  ]) / dx1[i];
        const real_t Ly = ((q[yp] - q[c]) / dmx2[j+1]
                         - (q[c] - q[ym]) / dmx2[j  ]) / dx2[j];
        // W is face-centered in z; U/V are cell-centered in z.
        const real_t Lz = z_staggered
            ? ((q[zp] - q[c]) / dx3[k  ]
             - (q[c] - q[zm]) / dx3[k-1]) / dmx3[k]
            : ((q[zp] - q[c]) / dmx3[k+1]
             - (q[c] - q[zm]) / dmx3[k  ]) / dx3[k];

        // Standard Laplacian contribution (all directions once).
        const real_t visc_rhs = Lx + Ly + Lz;

        // --- Convection: 0.25 * div(u*q) (conservative flux form) ---
        const real_t fx_p = U[xp] * real_t{0.5} * (q[c]  + q[xp]);
        const real_t fx_m = U[c ] * real_t{0.5} * (q[xm] + q[c ]);
        const real_t fy_p = V[yp] * real_t{0.5} * (q[c]  + q[yp]);
        const real_t fy_m = V[c ] * real_t{0.5} * (q[ym] + q[c ]);
        const real_t fz_p = W[zp] * real_t{0.5} * (q[c]  + q[zp]);
        const real_t fz_m = W[c ] * real_t{0.5} * (q[zm] + q[c ]);

        const real_t div_uq = (fx_p - fx_m) / dx1[i]
                            + (fy_p - fy_m) / dx2[j]
                            + (fz_p - fz_m) / dx3[k];

        rhs[c] = nu_dt * visc_rhs + conv_coeff * div_uq;
      }
}


// =============================================================================
//  ADI band assembly (MPM-STD BW-ADI)
//
//  Builds the tridiagonal system (I − visc_factor*dt*nu*L_d − 0.25*dt*conv_d)
//  along direction d.
//
//  visc_factor = 1.0 → backward Euler (own direction)
//  visc_factor = 0.5 → Crank-Nicolson (cross directions)
//
//  Convective bands: semi-implicit self-advection with 0.25 coefficient.
//  q_adv is the advecting velocity in direction d (U for X, V for Y, W for Z).
// =============================================================================
void build_adi_bands(Direction      d,
                      real_t*        A, real_t* B, real_t* C, real_t* D,
                      const real_t*  src,
                      const real_t*  q,
                      const real_t*  q_adv,
                      const real_t*  dx_along, const real_t* dmx_along,
                      int            n1_tot, int n2_tot, int n3_tot,
                      int            n1_int,  int n2_int,  int n3_int,
                      real_t         viscosity, real_t dt,
                      real_t         visc_factor,
                      bool           z_staggered) {
  // Viscous coefficient for this sweep.
  const real_t coeff = visc_factor * dt * viscosity;
  // Convective coefficient: 0.25
  const real_t cconv = real_t{0.25} * dt;

  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii) {
      const int    row = ii;
      const int    i   = ii + kHaloWidth;
      // Viscous stencil (cell-centered in x for U: dx_along=dx1, dmx_along=dmx1)
      const real_t av = -coeff / (dx_along[i] * dmx_along[i  ]);
      const real_t cv = -coeff / (dx_along[i] * dmx_along[i+1]);
      const real_t bv = real_t{1.0} - av - cv;
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = jj * n3_int + kk;
          const int p   = row * n_sys + sys;
          const int gi  = idx3(i, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gim = idx3(i-1, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gip = idx3(i+1, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);

          // Convective: u5 = 0.5*(q_adv[i-1]+q_adv[i]), u6 = 0.5*(q_adv[i]+q_adv[i+1])
          const real_t u5 = real_t{0.5} * (q_adv[gim] + q_adv[gi]);
          const real_t u6 = real_t{0.5} * (q_adv[gi]  + q_adv[gip]);
          const real_t ac = -cconv * u5 / dmx_along[i  ];
          const real_t cc =  cconv * u6 / dmx_along[i+1];
          const real_t bc = -cconv * (-u6 / dmx_along[i+1] + u5 / dmx_along[i]);

          A[p] = av + ac;
          B[p] = bv + bc;
          C[p] = cv + cc;
          D[p] = src[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                            n1_tot, n2_tot, n3_tot)];
        }
    }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int    row = jj;
      const int    j   = jj + kHaloWidth;
      const real_t av = -coeff / (dx_along[j] * dmx_along[j  ]);
      const real_t cv = -coeff / (dx_along[j] * dmx_along[j+1]);
      const real_t bv = real_t{1.0} - av - cv;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = ii * n3_int + kk;
          const int p   = row * n_sys + sys;
          const int gi  = idx3(ii+kHaloWidth, j, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gjm = idx3(ii+kHaloWidth, j-1, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gjp = idx3(ii+kHaloWidth, j+1, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);

          const real_t v5 = real_t{0.5} * (q_adv[gjm] + q_adv[gi]);
          const real_t v6 = real_t{0.5} * (q_adv[gi]  + q_adv[gjp]);
          const real_t ac = -cconv * v5 / dmx_along[j  ];
          const real_t cc =  cconv * v6 / dmx_along[j+1];
          const real_t bc = -cconv * (-v6 / dmx_along[j+1] + v5 / dmx_along[j]);

          A[p] = av + ac;
          B[p] = bv + bc;
          C[p] = cv + cc;
          D[p] = src[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                            n1_tot, n2_tot, n3_tot)];
        }
    }
  } else { // Direction::Z
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk) {
      const int    row = kk;
      const int    k   = kk + kHaloWidth;
      // W is face-centered in z: control volume = dmx[k], neighbor dist = dx[k-1], dx[k]
      // U/V cell-centered in z: control volume = dx[k], neighbor dist = dmx[k], dmx[k+1]
      const real_t av = z_staggered
          ? -coeff / (dmx_along[k] * dx_along[k-1])
          : -coeff / (dx_along[k]  * dmx_along[k  ]);
      const real_t cv = z_staggered
          ? -coeff / (dmx_along[k] * dx_along[k  ])
          : -coeff / (dx_along[k]  * dmx_along[k+1]);
      const real_t bv = real_t{1.0} - av - cv;

      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int sys = ii * n2_int + jj;
          const int p   = row * n_sys + sys;
          const int gi  = idx3(ii+kHaloWidth, jj+kHaloWidth, k,   n1_tot, n2_tot, n3_tot);
          const int gkm = idx3(ii+kHaloWidth, jj+kHaloWidth, k-1, n1_tot, n2_tot, n3_tot);
          const int gkp = idx3(ii+kHaloWidth, jj+kHaloWidth, k+1, n1_tot, n2_tot, n3_tot);

          // Face velocities for z-direction convection.
          const real_t w5 = real_t{0.5} * (q_adv[gkm] + q_adv[gi]);
          const real_t w6 = real_t{0.5} * (q_adv[gi]  + q_adv[gkp]);
          real_t ac, cc, bc;
          if (z_staggered) {
            // W face-centered in z: control-volume width = dmx3[k]
            ac = -cconv * w5 / dx_along[k-1];
            cc =  cconv * w6 / dx_along[k  ];
            bc = -cconv * (-w6 / dx_along[k] + w5 / dx_along[k-1]);
          } else {
            // U/V cell-centered in z: control-volume width = dx3[k]
            ac = -cconv * w5 / dmx_along[k  ];
            cc =  cconv * w6 / dmx_along[k+1];
            bc = -cconv * (-w6 / dmx_along[k+1] + w5 / dmx_along[k]);
          }

          A[p] = av + ac;
          B[p] = bv + bc;
          C[p] = cv + cc;
          D[p] = src[idx3(ii + kHaloWidth, jj + kHaloWidth, kk + kHaloWidth,
                            n1_tot, n2_tot, n3_tot)];
        }
    }
  }
}


// =============================================================================
//  (3) Source term accumulation: rhs[c] += dt * src[c]
// =============================================================================
void add_source_rhs(real_t*       rhs,
                     const real_t* src,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int c = idx3(i, j, k, n1_tot, n2_tot, n3_tot);
        rhs[c] += dt * src[c];
      }
}


// =============================================================================
//  (4) Uniform body force: rhs[c] += val (interior cells)
// =============================================================================
void add_constant_rhs(real_t* rhs,
                       int n1_tot, int n2_tot, int n3_tot,
                       real_t val) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k)
        rhs[idx3(i, j, k, n1_tot, n2_tot, n3_tot)] += val;
}


// =============================================================================
//  (5) Scatter TDMA solution back to 3-D buffer
// =============================================================================
void scatter_from_tdma(Direction      d,
                        real_t*        dst,
                        const real_t*  D_solution,
                        int            n1_tot, int n2_tot, int n3_tot,
                        int            n1_int,  int n2_int,  int n3_int) {
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
//  (6) q += delta (interior cells)
// =============================================================================
void add_increment(real_t*       q,
                    const real_t* delta,
                    int n1_tot, int n2_tot, int n3_tot) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int p = idx3(i, j, k, n1_tot, n2_tot, n3_tot);
        q[p] += delta[p];
      }
}


// =============================================================================
//  (7) MPM-STD block coupling — blockLdV  (constant viscosity, convection only)
//      dV[i,j,k] -= dt * 0.25 * ( dW·∂V/∂z )   interpolated to the V (y-face) cell
//      Mirrors cuda_momentum_blockLdV_kernel (M23·dW), dropping cross-stress.
// =============================================================================
void block_couple_dV(real_t*       dV,
                     const real_t* dW,
                     const real_t* V,
                     const real_t* dx2, const real_t* dmx2,
                     const real_t* dx3, const real_t* dmx3,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int jm = j - 1, km = k - 1, kp = k + 1;
        const int c   = idx3(i, j,  k,   n1_tot, n2_tot, n3_tot);
        const int cjm = idx3(i, jm, k,   n1_tot, n2_tot, n3_tot);
        const int ckm = idx3(i, j,  km,  n1_tot, n2_tot, n3_tot);
        const int ckp = idx3(i, j,  kp,  n1_tot, n2_tot, n3_tot);

        // dW interpolated to the V-location's lower/upper z-faces (y-stagger).
        const real_t dwm5 = real_t{0.5} * (dx2[jm] * dW[c]   + dx2[j] * dW[cjm]) / dmx2[j];
        const real_t dwm6 = real_t{0.5} * (dx2[jm] * dW[ckp] + dx2[j] * dW[idx3(i, jm, kp, n1_tot, n2_tot, n3_tot)]) / dmx2[j];

        const real_t dvdz5 = (V[c]   - V[ckm]) / dmx3[k];
        const real_t dvdz6 = (V[ckp] - V[c])   / dmx3[kp];

        const real_t M23dW = real_t{0.25} * (dwm5 * dvdz5 + dwm6 * dvdz6);
        dV[c] -= dt * M23dW;
      }
}


// =============================================================================
//  (8) MPM-STD block coupling — blockLdU  (constant viscosity, convection only)
//      dU[i,j,k] -= dt * 0.25 * ( dV·∂U/∂y + dW·∂U/∂z )  at the U (x-face) cell
//      Mirrors cuda_momentum_blockLdU_kernel (M12·dV + M13·dW), no cross-stress.
//      Must run AFTER block_couple_dV (and a dV halo refresh): uses corrected dV.
// =============================================================================
void block_couple_dU(real_t*       dU,
                     const real_t* dV,
                     const real_t* dW,
                     const real_t* U,
                     const real_t* dx1, const real_t* dmx1,
                     const real_t* dmx2,
                     const real_t* dmx3,
                     int n1_tot, int n2_tot, int n3_tot,
                     real_t dt) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int im = i - 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
        const int c    = idx3(i,  j,  k,  n1_tot, n2_tot, n3_tot);
        const int cim  = idx3(im, j,  k,  n1_tot, n2_tot, n3_tot);
        const int cjm  = idx3(i,  jm, k,  n1_tot, n2_tot, n3_tot);
        const int cjp  = idx3(i,  jp, k,  n1_tot, n2_tot, n3_tot);
        const int ckm  = idx3(i,  j,  km, n1_tot, n2_tot, n3_tot);
        const int ckp  = idx3(i,  j,  kp, n1_tot, n2_tot, n3_tot);

        // dV interpolated to the U-location's lower/upper y-faces (x-stagger).
        const real_t dvm3 = real_t{0.5} * (dx1[im] * dV[c]   + dx1[i] * dV[cim]) / dmx1[i];
        const real_t dvm4 = real_t{0.5} * (dx1[im] * dV[cjp] + dx1[i] * dV[idx3(im, jp, k, n1_tot, n2_tot, n3_tot)]) / dmx1[i];
        const real_t dudy3 = (U[c]   - U[cjm]) / dmx2[j];
        const real_t dudy4 = (U[cjp] - U[c])   / dmx2[jp];

        // dW interpolated to the U-location's lower/upper z-faces (x-stagger).
        const real_t dwm5 = real_t{0.5} * (dx1[im] * dW[c]   + dx1[i] * dW[cim]) / dmx1[i];
        const real_t dwm6 = real_t{0.5} * (dx1[im] * dW[ckp] + dx1[i] * dW[idx3(im, j, kp, n1_tot, n2_tot, n3_tot)]) / dmx1[i];
        const real_t dudz5 = (U[c]   - U[ckm]) / dmx3[k];
        const real_t dudz6 = (U[ckp] - U[c])   / dmx3[kp];

        const real_t M12dV = real_t{0.25} * (dvm3 * dudy3 + dvm4 * dudy4);
        const real_t M13dW = real_t{0.25} * (dwm5 * dudz5 + dwm6 * dudz6);
        dU[c] -= dt * (M12dV + M13dW);
      }
}

} // namespace mpmstd::equation::momentum::kernels
