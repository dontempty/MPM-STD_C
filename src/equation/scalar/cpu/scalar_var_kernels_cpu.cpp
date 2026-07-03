#include "equation/scalar/kernels.hpp"
#include "common/macros.hpp"

// =============================================================================
//  Variable-κ̂ (NOB) ENERGY kernels — FAITHFUL line-by-line port of
//  PaScaL_TCS module_solve_thermal.f90 (mpi_thermal_solver). See kernels.hpp
//  for the scheme overview. This is the reference's delta-form 3-factor ADI:
//  the explicit residual R uses the 0.5-skew ADVECTIVE convection + variable-κ̂
//  diffusion with dκ̂ product-rule linearization + the dinvρĈp prefactor
//  linearization; each ADI band carries that direction's diffusion + dκ̂ + the
//  0.25 conservative-face SEMI-IMPLICIT convection (the upgrade that relaxes the
//  convective CFL at high Pr).
//
//  Reference axis x2 = wall-normal (velocity V) → my Z (velocity W); ref x3 → my
//  Y (velocity V); ref x1 → my X (velocity U). Because the per-axis operator is
//  structurally identical, each of my axes uses its own grid metrics + own face
//  velocity and the wall lives on Z. The reference's jem/jep wall masks + ebc
//  injection are NOT transcribed: this port keeps the validated mpmstd wall
//  wiring (antisymmetric θ ghost + modify_tdma_row_cpu fold), so δθ_wall=0 and
//  the wall coupling is dropped by the fold, not by a band mask.
// =============================================================================

namespace mpmstd::equation::scalar::kernels {

namespace {
MPMSTD_HD int idx3(int i, int j, int k, int /*n1*/, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

// Per-cell face quantities for ONE axis with neighbours (m=minus, p=plus):
// face-θ (e), face-θ-grad (ded), face-κ̂ (kPP), face-dκ̂ (dkPP) on the lower (lo)
// and upper (hi) faces — ref e1/e2, dedx1/dedx2, kPP1/kPP2, dkPP1/dkPP2 etc.
// Volume-weighted (dx-weighted) face interpolation, exactly as the reference.
struct FaceStencil {
  real_t e_lo, e_hi;        // ref e1,e2     : face θ                (ref 257-262)
  real_t ded_lo, ded_hi;    // ref dedx1,2   : face θ-gradient       (ref 263-268)
  real_t kPP_lo, kPP_hi;    // ref kPP1,2    : face κ̂               (ref 277-282)
  real_t dkPP_lo, dkPP_hi;  // ref dkPP1,2   : face dκ̂              (ref 284-289)
};

// cm/cc/cp = flat indices of minus/centre/plus cells along the axis.
// dx_m = dx[idx-1], dx_c = dx[idx], dx_p = dx[idx+1];
// dmx_lo = dmx[idx] (lower face), dmx_hi = dmx[idx+1] (upper face).
MPMSTD_HD FaceStencil face_stencil(const real_t* T, const real_t* kappa, const real_t* dkappa,
                                   int cm, int cc, int cp,
                                   real_t dx_m, real_t dx_c, real_t dx_p,
                                   real_t dmx_lo, real_t dmx_hi) {
  FaceStencil f;
  // e1 = (dx(i)·θ(im) + dx(im)·θ(i)) · 0.5/dmx(i)      [ref 257]
  f.e_lo = (dx_c * T[cm] + dx_m * T[cc]) * (real_t{0.5} / dmx_lo);
  // e2 = (dx(ip)·θ(i)  + dx(i)·θ(ip)) · 0.5/dmx(ip)    [ref 258]
  f.e_hi = (dx_p * T[cc] + dx_c * T[cp]) * (real_t{0.5} / dmx_hi);
  f.ded_lo = (T[cc] - T[cm]) / dmx_lo;                 // dedx1 [ref 263]
  f.ded_hi = (T[cp] - T[cc]) / dmx_hi;                 // dedx2 [ref 264]
  // kPP1 = 0.5/dmx(i)·(dx(i)·κ̂(im) + dx(im)·κ̂(i))     [ref 277]
  f.kPP_lo = (real_t{0.5} / dmx_lo) * (dx_c * kappa[cm] + dx_m * kappa[cc]);
  f.kPP_hi = (real_t{0.5} / dmx_hi) * (dx_p * kappa[cc] + dx_c * kappa[cp]);   // [ref 278]
  f.dkPP_lo = (real_t{0.5} / dmx_lo) * (dx_c * dkappa[cm] + dx_m * dkappa[cc]); // [ref 284]
  f.dkPP_hi = (real_t{0.5} / dmx_hi) * (dx_p * dkappa[cc] + dx_c * dkappa[cp]); // [ref 285]
  return f;
}

// Bare flux divergence ∇·(κ̂∇θ) along one axis (no dκ̂ correction): the
// (kPP_hi·ded_hi − kPP_lo·ded_lo)/dx_c that the prefactor linearization uses
// (ref's (kPP2·dedx2 − kPP1·dedx1)/dx1, etc.).
MPMSTD_HD real_t div_bare(const FaceStencil& f, real_t dx_c) {
  return (f.kPP_hi * f.ded_hi - f.kPP_lo * f.ded_lo) / dx_c;
}

// Per-axis ADI band coefficients (ref X:324-329, the generic block):
//   eAC = irc_h/dx·( kPP_hi/dmx_hi + kPP_lo/dmx_lo
//                    − dx_p·(0.5/dmx_hi)·dkPP_hi·ded_hi
//                    + dx_m·(0.5/dmx_lo)·dkPP_lo·ded_lo )
//              + (0.25·vel_lo/dmx_lo − 0.25·vel_hi/dmx_hi)
//   eAP = −irc_h/dx·( kPP_hi/dmx_hi + dx_c·(0.5/dmx_hi)·dkPP_hi·ded_hi ) + 0.25·vel_hi/dmx_hi
//   eAM = −irc_h/dx·( kPP_lo/dmx_lo − dx_c·(0.5/dmx_lo)·dkPP_lo·ded_lo ) − 0.25·vel_lo/dmx_lo
// (No jem/jep masks — wall handled by ghost + modify_tdma_row_cpu.)
MPMSTD_HD void axis_bands(const FaceStencil& f, real_t irc_h,
                          real_t dx_m, real_t dx_c, real_t dx_p,
                          real_t dmx_lo, real_t dmx_hi, real_t vel_lo, real_t vel_hi,
                          real_t& eAM, real_t& eAC, real_t& eAP) {
  eAC = irc_h / dx_c * ( f.kPP_hi / dmx_hi + f.kPP_lo / dmx_lo
                       - dx_p * (real_t{0.5}/dmx_hi) * f.dkPP_hi * f.ded_hi
                       + dx_m * (real_t{0.5}/dmx_lo) * f.dkPP_lo * f.ded_lo )
      + (real_t{0.25}*vel_lo/dmx_lo - real_t{0.25}*vel_hi/dmx_hi);
  eAP = -irc_h / dx_c * ( f.kPP_hi / dmx_hi + dx_c*(real_t{0.5}/dmx_hi)*f.dkPP_hi*f.ded_hi )
        + real_t{0.25}*vel_hi/dmx_hi;
  eAM = -irc_h / dx_c * ( f.kPP_lo / dmx_lo - dx_c*(real_t{0.5}/dmx_lo)*f.dkPP_lo*f.ded_lo )
        - real_t{0.25}*vel_lo/dmx_lo;
}
} // anonymous namespace


// =============================================================================
//  Explicit residual R  (ref 256-299, the j/k loop body before the bands)
// =============================================================================
void energy_residual_rhs_var(real_t*       rhs,
                             const real_t* T,
                             const real_t* U, const real_t* V, const real_t* W,
                             const real_t* kappa, const real_t* dkappa,
                             const real_t* inv_rhocp, const real_t* dinv_rhocp,
                             const real_t* dx1, const real_t* dmx1,
                             const real_t* dx2, const real_t* dmx2,
                             const real_t* dx3, const real_t* dmx3,
                             int n1_tot, int n2_tot, int n3_tot,
                             real_t Ct) {
  const int i0 = kHaloWidth, i1 = n1_tot - kHaloWidth;
  const int j0 = kHaloWidth, j1 = n2_tot - kHaloWidth;
  const int k0 = kHaloWidth, k1 = n3_tot - kHaloWidth;

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

        const real_t irc_h  = real_t{0.5} * Ct * inv_rhocp[c];   // ref 253
        const real_t dirc_h = real_t{0.5} * Ct * dinv_rhocp[c];  // ref 254

        const FaceStencil fx = face_stencil(T, kappa, dkappa, xm, c, xp,
                                            dx1[i-1], dx1[i], dx1[i+1], dmx1[i], dmx1[i+1]);
        const FaceStencil fy = face_stencil(T, kappa, dkappa, ym, c, yp,
                                            dx2[j-1], dx2[j], dx2[j+1], dmx2[j], dmx2[j+1]);
        const FaceStencil fz = face_stencil(T, kappa, dkappa, zm, c, zp,
                                            dx3[k-1], dx3[k], dx3[k+1], dmx3[k], dmx3[k+1]);

        // 0.5-skew advective convection  (ref 271-274): −0.5·Σ 0.5(vel_lo·ded_lo+vel_hi·ded_hi)
        const real_t conv_x = (U[c]  * fx.ded_lo + U[xp] * fx.ded_hi) * real_t{0.5};
        const real_t conv_y = (V[c]  * fy.ded_lo + V[yp] * fy.ded_hi) * real_t{0.5};
        const real_t conv_z = (W[c]  * fz.ded_lo + W[zp] * fz.ded_hi) * real_t{0.5};
        real_t R = -real_t{0.5} * (conv_x + conv_y + conv_z);

        // variable-κ̂ diffusion with dκ̂ product-rule linearization  (ref 291-293)
        const real_t ve_x = (fx.kPP_hi*fx.ded_hi - fx.kPP_lo*fx.ded_lo
                           - fx.e_hi*fx.dkPP_hi*fx.ded_hi + fx.e_lo*fx.dkPP_lo*fx.ded_lo) / dx1[i];
        const real_t ve_y = (fy.kPP_hi*fy.ded_hi - fy.kPP_lo*fy.ded_lo
                           - fy.e_hi*fy.dkPP_hi*fy.ded_hi + fy.e_lo*fy.dkPP_lo*fy.ded_lo) / dx2[j];
        const real_t ve_z = (fz.kPP_hi*fz.ded_hi - fz.kPP_lo*fz.ded_lo
                           - fz.e_hi*fz.dkPP_hi*fz.ded_hi + fz.e_lo*fz.dkPP_lo*fz.ded_lo) / dx3[k];

        const real_t div_sum = div_bare(fx, dx1[i]) + div_bare(fy, dx2[j]) + div_bare(fz, dx3[k]);

        // RHS += irc_h·Σve − dirc_h·θ·div_sum   (ref 295-299)
        R += irc_h * (ve_x + ve_y + ve_z) - dirc_h * T[c] * div_sum;

        rhs[c] = R;
      }
}


// =============================================================================
//  ADI band builder + delta-form RHS pack  (ref 316-358 stage1, 414-426 stage2,
//  489-504 stage3). One generic axis; the stage-1 vs stage-2/3 prefactor-lin
//  coefficient asymmetry (dirc_h vs irc_h, ref quirk) is the is_stage1 flag.
// =============================================================================
void build_energy_adi_bands_var(Direction      d,
                                real_t*        A, real_t* B, real_t* C, real_t* D,
                                const real_t*  src,
                                const real_t*  T,
                                const real_t*  U, const real_t* V, const real_t* W,
                                const real_t*  kappa, const real_t* dkappa,
                                const real_t*  inv_rhocp, const real_t* dinv_rhocp,
                                const real_t*  dx1, const real_t* dmx1,
                                const real_t*  dx2, const real_t* dmx2,
                                const real_t*  dx3, const real_t* dmx3,
                                int            n1_tot, int n2_tot, int n3_tot,
                                int            n1_int, int n2_int, int n3_int,
                                real_t         Ct, real_t dt,
                                bool           is_stage1) {
  // Compute the sweep-axis bands (am,ac,ap) and the packed RHS Dval at (i,j,k).
  // z_wall_lo / z_wall_hi: PaScaL_TCS jem/jep mask — zero dkPP at wall-adjacent
  // Z faces so the wall κ̂-gradient correction is dropped on the implicit side,
  // matching module_solve_thermal.f90 Stage 3 lines 490-491 (dble(jep/jem)=0).
  auto build_cell = [&](int i, int j, int k,
                        bool z_wall_lo, bool z_wall_hi,
                        real_t& am, real_t& ac, real_t& ap, real_t& Dval) {
    const int c  = idx3(i,   j,   k,   n1_tot, n2_tot, n3_tot);
    const int xm = idx3(i-1, j,   k,   n1_tot, n2_tot, n3_tot);
    const int xp = idx3(i+1, j,   k,   n1_tot, n2_tot, n3_tot);
    const int ym = idx3(i,   j-1, k,   n1_tot, n2_tot, n3_tot);
    const int yp = idx3(i,   j+1, k,   n1_tot, n2_tot, n3_tot);
    const int zm = idx3(i,   j,   k-1, n1_tot, n2_tot, n3_tot);
    const int zp = idx3(i,   j,   k+1, n1_tot, n2_tot, n3_tot);

    const real_t irc_h  = real_t{0.5} * Ct * inv_rhocp[c];   // ref 253/395/469
    const real_t dirc_h = real_t{0.5} * Ct * dinv_rhocp[c];  // ref 254 (stage1)

    const FaceStencil fx = face_stencil(T, kappa, dkappa, xm, c, xp,
                                        dx1[i-1], dx1[i], dx1[i+1], dmx1[i], dmx1[i+1]);
    const FaceStencil fy = face_stencil(T, kappa, dkappa, ym, c, yp,
                                        dx2[j-1], dx2[j], dx2[j+1], dmx2[j], dmx2[j+1]);
    FaceStencil fz = face_stencil(T, kappa, dkappa, zm, c, zp,
                                  dx3[k-1], dx3[k], dx3[k+1], dmx3[k], dmx3[k+1]);
    if (z_wall_lo) fz.dkPP_lo = real_t{0};   // PaScaL jem=0: drop dκ̂ at lower wall face
    if (z_wall_hi) fz.dkPP_hi = real_t{0};   // PaScaL jep=0: drop dκ̂ at upper wall face

    real_t eAMx,eACx,eAPx, eAMy,eACy,eAPy, eAMz,eACz,eAPz;
    axis_bands(fx, irc_h, dx1[i-1],dx1[i],dx1[i+1], dmx1[i],dmx1[i+1], U[c],U[xp], eAMx,eACx,eAPx);
    axis_bands(fy, irc_h, dx2[j-1],dx2[j],dx2[j+1], dmx2[j],dmx2[j+1], V[c],V[yp], eAMy,eACy,eAPy);
    axis_bands(fz, irc_h, dx3[k-1],dx3[k],dx3[k+1], dmx3[k],dmx3[k+1], W[c],W[zp], eAMz,eACz,eAPz);
    // PaScaL jep/jem full mask: zero complete Z off-diagonal at wall cells for
    // Stages 2&3 only (Stage 1 needs eAPz/eAMz in M·θⁿ to encode wall BC via ghost).
    if (!is_stage1) {
      if (z_wall_lo) eAMz = real_t{0};
      if (z_wall_hi) eAPz = real_t{0};
    }

    const real_t div_sum = div_bare(fx, dx1[i]) + div_bare(fy, dx2[j]) + div_bare(fz, dx3[k]);

    // select the sweep-axis bands
    real_t eAM_d, eAC_d, eAP_d;
    if      (d == Direction::X) { eAM_d=eAMx; eAC_d=eACx; eAP_d=eAPx; }
    else if (d == Direction::Y) { eAM_d=eAMy; eAC_d=eACy; eAP_d=eAPy; }
    else                        { eAM_d=eAMz; eAC_d=eACz; eAP_d=eAPz; }

    if (is_stage1) {
      // Stage 1: ac = eAC_d·dt − 0.5·dirc_h·div_sum·dt + 1   (ref 351-355, dinvRhoCp)
      ac = eAC_d * dt - real_t{0.5} * dirc_h * div_sum * dt + real_t{1};
      ap = eAP_d * dt;
      am = eAM_d * dt;
      // D = dt·( R − (Mx+My+Mz)·θⁿ + dirc_h·θⁿ·div_sum )   (ref 342-358)
      const real_t MxTn = eAPx*T[xp] + eACx*T[c] + eAMx*T[xm];
      const real_t MyTn = eAPy*T[yp] + eACy*T[c] + eAMy*T[ym];
      const real_t MzTn = eAPz*T[zp] + eACz*T[c] + eAMz*T[zm];
      const real_t R = src[c];   // R from energy_residual_rhs_var
      Dval = ( R - (MxTn + MyTn + MzTn) + dirc_h * T[c] * div_sum ) * dt;
    } else {
      // Stages 2 & 3: ac = (eAC_d − 0.5·invRhoCp·div_sum)·dt + 1
      //   (ref 418/493 — note: invRhoCpCt_half, NOT dinvRhoCp; nonzero at OB,
      //    a faithful reference asymmetry; zero at the conductive steady state).
      ac = (eAC_d - real_t{0.5} * irc_h * div_sum) * dt + real_t{1};
      ap = eAP_d * dt;
      am = eAM_d * dt;
      Dval = src[c];   // transposed running increment, copied through (ref 376/444)
    }
  };

  // dispatch over the sweep axis (same (sys,row) packing as the const path)
  // z_wall_lo/hi flags are carried into ALL stages so that Stage-1 M·θⁿ packing
  // and Stage-3 implicit bands both use the same masked fz at wall-adjacent cells.
  if (d == Direction::X) {
    const int n_sys = n2_int * n3_int;
    for (int ii = 0; ii < n1_int; ++ii) {
      const int i = ii + kHaloWidth;
      for (int jj = 0; jj < n2_int; ++jj)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int j = jj + kHaloWidth, k = kk + kHaloWidth;
          const int p = ii * n_sys + (jj * n3_int + kk);
          real_t am, ac, ap, Dv;
          build_cell(i, j, k, kk==0, kk==n3_int-1, am, ac, ap, Dv);
          A[p]=am; B[p]=ac; C[p]=ap; D[p]=Dv;
        }
    }
  } else if (d == Direction::Y) {
    const int n_sys = n1_int * n3_int;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int j = jj + kHaloWidth;
      for (int ii = 0; ii < n1_int; ++ii)
        for (int kk = 0; kk < n3_int; ++kk) {
          const int i = ii + kHaloWidth, k = kk + kHaloWidth;
          const int p = jj * n_sys + (ii * n3_int + kk);
          real_t am, ac, ap, Dv;
          build_cell(i, j, k, kk==0, kk==n3_int-1, am, ac, ap, Dv);
          A[p]=am; B[p]=ac; C[p]=ap; D[p]=Dv;
        }
    }
  } else { // Direction::Z  (wall-normal) — PaScaL Stage 3 with jem/jep dkPP mask
    const int n_sys = n1_int * n2_int;
    for (int kk = 0; kk < n3_int; ++kk) {
      const int k = kk + kHaloWidth;
      const bool wall_lo = (kk == 0);
      const bool wall_hi = (kk == n3_int - 1);
      for (int ii = 0; ii < n1_int; ++ii)
        for (int jj = 0; jj < n2_int; ++jj) {
          const int i = ii + kHaloWidth, j = jj + kHaloWidth;
          const int p = kk * n_sys + (ii * n2_int + jj);
          real_t am, ac, ap, Dv; build_cell(i, j, k, wall_lo, wall_hi, am, ac, ap, Dv);
          A[p]=am; B[p]=ac; C[p]=ap; D[p]=Dv;
        }
    }
  }
}

} // namespace mpmstd::equation::scalar::kernels
