#include "equation/momentum/kernels.hpp"
#include "common/macros.hpp"

// Variable-viscosity (NOB) momentum kernels — face-interpolated μ + 1/ρ̂
// prefactor. Same structure as the const-visc kernels (momentum_kernels_cpu.cpp)
// so that with μ≡1, invrho≡1, Cmu=ν they reduce EXACTLY to them (the validation
// hook). Cross-stress terms are added by the caller (they vanish for constant μ
// via continuity → don't affect the μ≡1 match). Stage P7-step1: diagonal stress
// only; cross-stress is a follow-up refinement.

namespace mpmstd::equation::momentum::kernels {

namespace {
MPMSTD_HD int idx3v(int i, int j, int k, int /*n1*/, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}
} // anonymous namespace

void compute_mpmstd_rhs_var(real_t*       rhs,
                            const real_t* q,
                            const real_t* U, const real_t* V, const real_t* W,
                            const real_t* mu, const real_t* invrho,
                            const real_t* dx1, const real_t* dmx1,
                            const real_t* dx2, const real_t* dmx2,
                            const real_t* dx3, const real_t* dmx3,
                            int n1_tot, int n2_tot, int n3_tot,
                            real_t Cmu, real_t dt,
                            bool z_staggered,
                            real_t conv_factor) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;
  const real_t conv_coeff = -conv_factor * dt;
  const real_t half = real_t{0.5};

  for (int i = i0; i < i1; ++i)
    for (int j = j0; j < j1; ++j)
      for (int k = k0; k < k1; ++k) {
        const int c  = idx3v(i,   j,   k,   n1_tot, n2_tot, n3_tot);
        const int xm = idx3v(i-1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int xp = idx3v(i+1, j,   k,   n1_tot, n2_tot, n3_tot);
        const int ym = idx3v(i,   j-1, k,   n1_tot, n2_tot, n3_tot);
        const int yp = idx3v(i,   j+1, k,   n1_tot, n2_tot, n3_tot);
        const int zm = idx3v(i,   j,   k-1, n1_tot, n2_tot, n3_tot);
        const int zp = idx3v(i,   j,   k+1, n1_tot, n2_tot, n3_tot);

        // Face-interpolated μ (→ 1 when μ≡1).
        const real_t mxp = half * (mu[c] + mu[xp]), mxm = half * (mu[xm] + mu[c]);
        const real_t myp = half * (mu[c] + mu[yp]), mym = half * (mu[ym] + mu[c]);
        const real_t mzp = half * (mu[c] + mu[zp]), mzm = half * (mu[zm] + mu[c]);

        const real_t Lx = (mxp * (q[xp] - q[c]) / dmx1[i+1]
                         - mxm * (q[c] - q[xm]) / dmx1[i  ]) / dx1[i];
        const real_t Ly = (myp * (q[yp] - q[c]) / dmx2[j+1]
                         - mym * (q[c] - q[ym]) / dmx2[j  ]) / dx2[j];
        const real_t Lz = z_staggered
            ? (mzp * (q[zp] - q[c]) / dx3[k  ]
             - mzm * (q[c] - q[zm]) / dx3[k-1]) / dmx3[k]
            : (mzp * (q[zp] - q[c]) / dmx3[k+1]
             - mzm * (q[c] - q[zm]) / dmx3[k  ]) / dx3[k];

        const real_t visc = invrho[c] * Cmu * (Lx + Ly + Lz);

        // Convection — identical to the const-visc kernel.
        const real_t fx_p = U[xp] * half * (q[c]  + q[xp]);
        const real_t fx_m = U[c ] * half * (q[xm] + q[c ]);
        const real_t fy_p = V[yp] * half * (q[c]  + q[yp]);
        const real_t fy_m = V[c ] * half * (q[ym] + q[c ]);
        const real_t fz_p = W[zp] * half * (q[c]  + q[zp]);
        const real_t fz_m = W[c ] * half * (q[zm] + q[c ]);
        const real_t div_uq = (fx_p - fx_m) / dx1[i]
                            + (fy_p - fy_m) / dx2[j]
                            + (fz_p - fz_m) / dx3[k];

        rhs[c] = dt * visc + conv_coeff * div_uq;
      }
}

void build_adi_bands_var(Direction      d,
                         real_t*        A, real_t* B, real_t* C, real_t* D,
                         const real_t*  src,
                         const real_t*  /*q*/,
                         const real_t*  q_adv,
                         const real_t*  mu, const real_t* invrho,
                         const real_t*  dx_along, const real_t* dmx_along,
                         int            n1_tot, int n2_tot, int n3_tot,
                         int            n1_int,  int n2_int,  int n3_int,
                         real_t         Cmu, real_t dt,
                         real_t         visc_factor,
                         bool           z_staggered) {
  const real_t vbase = visc_factor * dt * Cmu;   // × invrho × μ_face per cell
  const real_t cconv = real_t{0.25} * dt;
  const real_t half  = real_t{0.5};

  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii) {
      const int row = ii, i = ii + kHaloWidth;
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = jj * n3_int + kk, p = row * n_sys + sys;
          const int gi  = idx3v(i,   jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gim = idx3v(i-1, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gip = idx3v(i+1, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const real_t ir = invrho[gi];
          const real_t mm = half * (mu[gim] + mu[gi]), mp = half * (mu[gi] + mu[gip]);
          const real_t av = -vbase * ir * mm / (dx_along[i] * dmx_along[i  ]);
          const real_t cv = -vbase * ir * mp / (dx_along[i] * dmx_along[i+1]);
          const real_t bv = real_t{1.0} - av - cv;
          const real_t u5 = half * (q_adv[gim] + q_adv[gi]);
          const real_t u6 = half * (q_adv[gi]  + q_adv[gip]);
          const real_t ac = -cconv * u5 / dmx_along[i  ];
          const real_t cc =  cconv * u6 / dmx_along[i+1];
          const real_t bc = -cconv * (-u6 / dmx_along[i+1] + u5 / dmx_along[i]);
          A[p] = av + ac; B[p] = bv + bc; C[p] = cv + cc;
          D[p] = src[idx3v(ii+kHaloWidth, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot)];
        }
    }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int row = jj, j = jj + kHaloWidth;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int sys = ii * n3_int + kk, p = row * n_sys + sys;
          const int gi  = idx3v(ii+kHaloWidth, j,   kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gjm = idx3v(ii+kHaloWidth, j-1, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const int gjp = idx3v(ii+kHaloWidth, j+1, kk+kHaloWidth, n1_tot, n2_tot, n3_tot);
          const real_t ir = invrho[gi];
          const real_t mm = half * (mu[gjm] + mu[gi]), mp = half * (mu[gi] + mu[gjp]);
          const real_t av = -vbase * ir * mm / (dx_along[j] * dmx_along[j  ]);
          const real_t cv = -vbase * ir * mp / (dx_along[j] * dmx_along[j+1]);
          const real_t bv = real_t{1.0} - av - cv;
          const real_t v5 = half * (q_adv[gjm] + q_adv[gi]);
          const real_t v6 = half * (q_adv[gi]  + q_adv[gjp]);
          const real_t ac = -cconv * v5 / dmx_along[j  ];
          const real_t cc =  cconv * v6 / dmx_along[j+1];
          const real_t bc = -cconv * (-v6 / dmx_along[j+1] + v5 / dmx_along[j]);
          A[p] = av + ac; B[p] = bv + bc; C[p] = cv + cc;
          D[p] = src[idx3v(ii+kHaloWidth, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot)];
        }
    }
  } else { // Direction::Z
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk) {
      const int row = kk, k = kk + kHaloWidth;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int sys = ii * n2_int + jj, p = row * n_sys + sys;
          const int gi  = idx3v(ii+kHaloWidth, jj+kHaloWidth, k,   n1_tot, n2_tot, n3_tot);
          const int gkm = idx3v(ii+kHaloWidth, jj+kHaloWidth, k-1, n1_tot, n2_tot, n3_tot);
          const int gkp = idx3v(ii+kHaloWidth, jj+kHaloWidth, k+1, n1_tot, n2_tot, n3_tot);
          const real_t ir = invrho[gi];
          const real_t mm = half * (mu[gkm] + mu[gi]), mp = half * (mu[gi] + mu[gkp]);
          const real_t av = z_staggered ? -vbase * ir * mm / (dmx_along[k] * dx_along[k-1])
                                         : -vbase * ir * mm / (dx_along[k]  * dmx_along[k  ]);
          const real_t cv = z_staggered ? -vbase * ir * mp / (dmx_along[k] * dx_along[k  ])
                                         : -vbase * ir * mp / (dx_along[k]  * dmx_along[k+1]);
          const real_t bv = real_t{1.0} - av - cv;
          const real_t w5 = half * (q_adv[gkm] + q_adv[gi]);
          const real_t w6 = half * (q_adv[gi]  + q_adv[gkp]);
          real_t ac, cc, bc;
          if (z_staggered) {
            ac = -cconv * w5 / dx_along[k-1];
            cc =  cconv * w6 / dx_along[k  ];
            bc = -cconv * (-w6 / dx_along[k] + w5 / dx_along[k-1]);
          } else {
            ac = -cconv * w5 / dmx_along[k  ];
            cc =  cconv * w6 / dmx_along[k+1];
            bc = -cconv * (-w6 / dmx_along[k+1] + w5 / dmx_along[k]);
          }
          A[p] = av + ac; B[p] = bv + bc; C[p] = cv + cc;
          D[p] = src[idx3v(ii+kHaloWidth, jj+kHaloWidth, kk+kHaloWidth, n1_tot, n2_tot, n3_tot)];
        }
    }
  }
}

} // namespace mpmstd::equation::momentum::kernels
