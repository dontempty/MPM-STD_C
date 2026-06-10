#include "equation/scalar/kernels.hpp"
#include "common/macros.hpp"

// Variable-conductivity (NOB) scalar kernels — the κ̂(θ)/1/(ρ̂Ĉp) twins of the
// const versions in scalar_kernels_cpu.cpp. The operator is
//     (1/(ρ̂Ĉp)) ∇·(κ̂ ∇θ)              scaled by Ct = 1/√(Ra·Pr),
// with face-interpolated conductivity κ̂_{i+1/2}=0.5(κ̂_i+κ̂_{i+1}). With κ̂≡1,
// invρĈp≡1, Ct=alpha these reduce bit-for-bit to the const kernels (face-interp
// of a constant is the constant; the cell prefactor is 1) — the OB limit. No
// dκ̂ product-rule linearisation (the stiff-Pr stability refinement) yet.

namespace mpmstd::equation::scalar::kernels {

namespace {
MPMSTD_HD int idx3(int i, int j, int k, int /*n1*/, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}
} // anonymous namespace

// =============================================================================
//  Explicit variable-κ̂ Laplacian:  rhs = Ct*dt*(1/(ρ̂Ĉp))_c * ∇·(κ̂ ∇θ)
// =============================================================================
void laplacian_explicit_rhs_var(real_t*       rhs,
                                const real_t* phi,
                                const real_t* kappa, const real_t* inv_rhocp,
                                const real_t* dx1, const real_t* dmx1,
                                const real_t* dx2, const real_t* dmx2,
                                const real_t* dx3, const real_t* dmx3,
                                int n1_tot, int n2_tot, int n3_tot,
                                real_t Ct, real_t dt) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;
  const real_t Ct_dt = Ct * dt;

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

        // Face-interpolated conductivities.
        const real_t kc  = kappa[c];
        const real_t kxm = static_cast<real_t>(0.5) * (kc + kappa[xm]);
        const real_t kxp = static_cast<real_t>(0.5) * (kc + kappa[xp]);
        const real_t kym = static_cast<real_t>(0.5) * (kc + kappa[ym]);
        const real_t kyp = static_cast<real_t>(0.5) * (kc + kappa[yp]);
        const real_t kzm = static_cast<real_t>(0.5) * (kc + kappa[zm]);
        const real_t kzp = static_cast<real_t>(0.5) * (kc + kappa[zp]);

        // ∇·(κ̂∇θ) per axis:  (1/dx_i)[ κ̂_{i+1/2}(φ_{i+1}-φ_i)/dmx_{i+1}
        //                              - κ̂_{i-1/2}(φ_i-φ_{i-1})/dmx_i ]
        const real_t Lx = (kxp * (phi[xp] - phi[c]) / dmx1[i+1]
                         - kxm * (phi[c]  - phi[xm]) / dmx1[i  ]) / dx1[i];
        const real_t Ly = (kyp * (phi[yp] - phi[c]) / dmx2[j+1]
                         - kym * (phi[c]  - phi[ym]) / dmx2[j  ]) / dx2[j];
        const real_t Lz = (kzp * (phi[zp] - phi[c]) / dmx3[k+1]
                         - kzm * (phi[c]  - phi[zm]) / dmx3[k  ]) / dx3[k];

        rhs[c] = Ct_dt * inv_rhocp[c] * (Lx + Ly + Lz);
      }
}

// =============================================================================
//  ADI bands for ONE axis with face-κ̂ + cell 1/(ρ̂Ĉp).  Implicit matrix is
//     (I - 0.5*dt*Ct*(1/(ρ̂Ĉp))_c * L_d^{κ̂}).  Coefficients vary per cell, so
//  the bands are filled per (row, sys) rather than per row.
//     A_i = -0.5*dt*Ct*ir_c * κ̂_{i-1/2} / (dx_i * dmx_i)        (sub)
//     C_i = -0.5*dt*Ct*ir_c * κ̂_{i+1/2} / (dx_i * dmx_{i+1})    (super)
//     B_i =  1 - A_i - C_i                                       (diag)
// =============================================================================
void build_adi_bands_var(Direction      d,
                         real_t*        A, real_t* B, real_t* C, real_t* D,
                         const real_t*  src,
                         const real_t*  kappa, const real_t* inv_rhocp,
                         const real_t*  dx_along, const real_t* dmx_along,
                         int            n1_tot, int n2_tot, int n3_tot,
                         int            n1_int, int n2_int, int n3_int,
                         real_t         Ct, real_t dt) {
  const real_t half_Ct_dt = static_cast<real_t>(0.5) * Ct * dt;

  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii) {
      const int i = ii + kHaloWidth;
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int j = jj + kHaloWidth, k = kk + kHaloWidth;
          const int c  = idx3(i,   j, k, n1_tot, n2_tot, n3_tot);
          const int xm = idx3(i-1, j, k, n1_tot, n2_tot, n3_tot);
          const int xp = idx3(i+1, j, k, n1_tot, n2_tot, n3_tot);
          const real_t ir  = inv_rhocp[c];
          const real_t kfm = static_cast<real_t>(0.5) * (kappa[c] + kappa[xm]);
          const real_t kfp = static_cast<real_t>(0.5) * (kappa[c] + kappa[xp]);
          const real_t a = -half_Ct_dt * ir * kfm / (dx_along[i] * dmx_along[i  ]);
          const real_t cc= -half_Ct_dt * ir * kfp / (dx_along[i] * dmx_along[i+1]);
          const int sys = jj * n3_int + kk;
          const int p   = ii * n_sys + sys;
          A[p] = a;  C[p] = cc;  B[p] = static_cast<real_t>(1.0) - a - cc;  D[p] = src[c];
        }
    }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int j = jj + kHaloWidth;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int i = ii + kHaloWidth, k = kk + kHaloWidth;
          const int c  = idx3(i, j,   k, n1_tot, n2_tot, n3_tot);
          const int ym = idx3(i, j-1, k, n1_tot, n2_tot, n3_tot);
          const int yp = idx3(i, j+1, k, n1_tot, n2_tot, n3_tot);
          const real_t ir  = inv_rhocp[c];
          const real_t kfm = static_cast<real_t>(0.5) * (kappa[c] + kappa[ym]);
          const real_t kfp = static_cast<real_t>(0.5) * (kappa[c] + kappa[yp]);
          const real_t a = -half_Ct_dt * ir * kfm / (dx_along[j] * dmx_along[j  ]);
          const real_t cc= -half_Ct_dt * ir * kfp / (dx_along[j] * dmx_along[j+1]);
          const int sys = ii * n3_int + kk;
          const int p   = jj * n_sys + sys;
          A[p] = a;  C[p] = cc;  B[p] = static_cast<real_t>(1.0) - a - cc;  D[p] = src[c];
        }
    }
  } else { // Direction::Z
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk) {
      const int k = kk + kHaloWidth;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int i = ii + kHaloWidth, j = jj + kHaloWidth;
          const int c  = idx3(i, j, k,   n1_tot, n2_tot, n3_tot);
          const int zm = idx3(i, j, k-1, n1_tot, n2_tot, n3_tot);
          const int zp = idx3(i, j, k+1, n1_tot, n2_tot, n3_tot);
          const real_t ir  = inv_rhocp[c];
          const real_t kfm = static_cast<real_t>(0.5) * (kappa[c] + kappa[zm]);
          const real_t kfp = static_cast<real_t>(0.5) * (kappa[c] + kappa[zp]);
          const real_t a = -half_Ct_dt * ir * kfm / (dx_along[k] * dmx_along[k  ]);
          const real_t cc= -half_Ct_dt * ir * kfp / (dx_along[k] * dmx_along[k+1]);
          const int sys = ii * n2_int + jj;
          const int p   = kk * n_sys + sys;
          A[p] = a;  C[p] = cc;  B[p] = static_cast<real_t>(1.0) - a - cc;  D[p] = src[c];
        }
    }
  }
}

} // namespace mpmstd::equation::scalar::kernels
