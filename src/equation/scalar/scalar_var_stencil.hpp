#pragma once

// MPMSTD_HD helpers for the variable-κ̂ (NOB) scalar stencil.
// Included by both scalar_var_kernels_cpu.cpp and scalar_var_kernels_gpu.cu.
// All functions are __host__ __device__ (via MPMSTD_HD) so they compile for
// both CPU and device translation units.

#include "common/macros.hpp"
#include "common/types.hpp"
#include "common/direction.hpp"

namespace mpmstd::equation::scalar::kernels {
namespace stencil {

static constexpr int kH = 1;  // halo width

MPMSTD_HD int idx3(int i, int j, int k, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

// Face values of T, κ̂, dκ̂ along ONE axis at a given cell.
// Matches PaScaL_TCS ref 257-289.
struct FaceStencil {
  real_t e_lo, e_hi;
  real_t ded_lo, ded_hi;
  real_t kPP_lo, kPP_hi;
  real_t dkPP_lo, dkPP_hi;
};

MPMSTD_HD FaceStencil
face_stencil(const real_t* T, const real_t* kappa, const real_t* dkappa,
             int cm, int cc, int cp,
             real_t dx_m, real_t dx_c, real_t dx_p,
             real_t dmx_lo, real_t dmx_hi) {
  FaceStencil f;
  f.e_lo  = (dx_c * T[cm] + dx_m * T[cc]) * (real_t{0.5} / dmx_lo);
  f.e_hi  = (dx_p * T[cc] + dx_c * T[cp]) * (real_t{0.5} / dmx_hi);
  f.ded_lo = (T[cc] - T[cm]) / dmx_lo;
  f.ded_hi = (T[cp] - T[cc]) / dmx_hi;
  f.kPP_lo  = (real_t{0.5} / dmx_lo) * (dx_c * kappa[cm] + dx_m * kappa[cc]);
  f.kPP_hi  = (real_t{0.5} / dmx_hi) * (dx_p * kappa[cc] + dx_c * kappa[cp]);
  f.dkPP_lo = (real_t{0.5} / dmx_lo) * (dx_c * dkappa[cm] + dx_m * dkappa[cc]);
  f.dkPP_hi = (real_t{0.5} / dmx_hi) * (dx_p * dkappa[cc] + dx_c * dkappa[cp]);
  return f;
}

MPMSTD_HD real_t div_bare(const FaceStencil& f, real_t dx_c) {
  return (f.kPP_hi * f.ded_hi - f.kPP_lo * f.ded_lo) / dx_c;
}

MPMSTD_HD void
axis_bands(const FaceStencil& f, real_t irc_h,
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

// Per-cell RHS and band computation — called by both CPU and GPU kernels.
// Writes (am, ac, ap, Dval) for the sweep-axis (d) band at cell (i,j,k).
// z_wall_lo / z_wall_hi: PaScaL_TCS jem/jep mask — when set, fz.dkPP_{lo,hi}
// is zeroed so that wall-adjacent cells drop the dκ̂ correction from the
// implicit operator (matches module_solve_thermal.f90 Stage 3 lines 490-491).
MPMSTD_HD void
build_cell_var(Direction d,
               int i, int j, int k,
               int n1_tot, int n2_tot, int n3_tot,
               const real_t* src, const real_t* T,
               const real_t* U, const real_t* V, const real_t* W,
               const real_t* kappa, const real_t* dkappa,
               const real_t* inv_rhocp, const real_t* dinv_rhocp,
               const real_t* dx1, const real_t* dmx1,
               const real_t* dx2, const real_t* dmx2,
               const real_t* dx3, const real_t* dmx3,
               real_t Ct, real_t dt, bool is_stage1,
               bool z_wall_lo, bool z_wall_hi,
               real_t& am, real_t& ac, real_t& ap, real_t& Dval) {
  const int c  = idx3(i,  j,  k,  n2_tot, n3_tot);
  const int xm = idx3(i-1,j,  k,  n2_tot, n3_tot);
  const int xp = idx3(i+1,j,  k,  n2_tot, n3_tot);
  const int ym = idx3(i,j-1,  k,  n2_tot, n3_tot);
  const int yp = idx3(i,j+1,  k,  n2_tot, n3_tot);
  const int zm = idx3(i,  j,k-1,  n2_tot, n3_tot);
  const int zp = idx3(i,  j,k+1,  n2_tot, n3_tot);

  const real_t irc_h  = real_t{0.5} * Ct * inv_rhocp[c];
  const real_t dirc_h = real_t{0.5} * Ct * dinv_rhocp[c];

  const FaceStencil fx = face_stencil(T,kappa,dkappa,xm,c,xp,dx1[i-1],dx1[i],dx1[i+1],dmx1[i],dmx1[i+1]);
  const FaceStencil fy = face_stencil(T,kappa,dkappa,ym,c,yp,dx2[j-1],dx2[j],dx2[j+1],dmx2[j],dmx2[j+1]);
  FaceStencil fz = face_stencil(T,kappa,dkappa,zm,c,zp,dx3[k-1],dx3[k],dx3[k+1],dmx3[k],dmx3[k+1]);
  if (z_wall_lo) fz.dkPP_lo = real_t{0};
  if (z_wall_hi) fz.dkPP_hi = real_t{0};

  real_t eAMx,eACx,eAPx, eAMy,eACy,eAPy, eAMz,eACz,eAPz;
  axis_bands(fx,irc_h,dx1[i-1],dx1[i],dx1[i+1],dmx1[i],dmx1[i+1],U[c],U[xp],eAMx,eACx,eAPx);
  axis_bands(fy,irc_h,dx2[j-1],dx2[j],dx2[j+1],dmx2[j],dmx2[j+1],V[c],V[yp],eAMy,eACy,eAPy);
  axis_bands(fz,irc_h,dx3[k-1],dx3[k],dx3[k+1],dmx3[k],dmx3[k+1],W[c],W[zp],eAMz,eACz,eAPz);
  // PaScaL jep/jem full mask (module_solve_thermal.f90 lines 511-512): zero the
  // complete Z off-diagonal at wall cells for Stages 2&3 (not Stage 1, where
  // eAPz/eAMz are needed in M·θⁿ to encode the wall BC in D via the ghost).
  // With ap/am pre-zeroed here, modify_tdma_row folds 0 → no diagonal inflation.
  if (!is_stage1) {
    if (z_wall_lo) eAMz = real_t{0};
    if (z_wall_hi) eAPz = real_t{0};
  }

  const real_t div_sum = div_bare(fx,dx1[i]) + div_bare(fy,dx2[j]) + div_bare(fz,dx3[k]);

  real_t eAM_d, eAC_d, eAP_d;
  if      (d == Direction::X) { eAM_d=eAMx; eAC_d=eACx; eAP_d=eAPx; }
  else if (d == Direction::Y) { eAM_d=eAMy; eAC_d=eACy; eAP_d=eAPy; }
  else                        { eAM_d=eAMz; eAC_d=eACz; eAP_d=eAPz; }

  if (is_stage1) {
    ac = eAC_d * dt - real_t{0.5} * dirc_h * div_sum * dt + real_t{1};
    ap = eAP_d * dt;
    am = eAM_d * dt;
    const real_t MxTn = eAPx*T[xp] + eACx*T[c] + eAMx*T[xm];
    const real_t MyTn = eAPy*T[yp] + eACy*T[c] + eAMy*T[ym];
    const real_t MzTn = eAPz*T[zp] + eACz*T[c] + eAMz*T[zm];
    Dval = ( src[c] - (MxTn + MyTn + MzTn) + dirc_h * T[c] * div_sum ) * dt;
  } else {
    ac = (eAC_d - real_t{0.5} * irc_h * div_sum) * dt + real_t{1};
    ap = eAP_d * dt;
    am = eAM_d * dt;
    Dval = src[c];
  }
}

// Per-cell RHS for energy_residual_rhs_var (explicit part).
MPMSTD_HD real_t
energy_rhs_cell(int i, int j, int k,
                int n2_tot, int n3_tot,
                const real_t* T, const real_t* U, const real_t* V, const real_t* W,
                const real_t* kappa, const real_t* dkappa,
                const real_t* inv_rhocp, const real_t* dinv_rhocp,
                const real_t* dx1, const real_t* dmx1,
                const real_t* dx2, const real_t* dmx2,
                const real_t* dx3, const real_t* dmx3,
                real_t Ct) {
  const int c  = idx3(i,  j,  k,  n2_tot, n3_tot);
  const int xm = idx3(i-1,j,  k,  n2_tot, n3_tot);
  const int xp = idx3(i+1,j,  k,  n2_tot, n3_tot);
  const int ym = idx3(i,j-1,  k,  n2_tot, n3_tot);
  const int yp = idx3(i,j+1,  k,  n2_tot, n3_tot);
  const int zm = idx3(i,  j,k-1,  n2_tot, n3_tot);
  const int zp = idx3(i,  j,k+1,  n2_tot, n3_tot);

  const real_t irc_h  = real_t{0.5} * Ct * inv_rhocp[c];
  const real_t dirc_h = real_t{0.5} * Ct * dinv_rhocp[c];

  const FaceStencil fx = face_stencil(T,kappa,dkappa,xm,c,xp,dx1[i-1],dx1[i],dx1[i+1],dmx1[i],dmx1[i+1]);
  const FaceStencil fy = face_stencil(T,kappa,dkappa,ym,c,yp,dx2[j-1],dx2[j],dx2[j+1],dmx2[j],dmx2[j+1]);
  const FaceStencil fz = face_stencil(T,kappa,dkappa,zm,c,zp,dx3[k-1],dx3[k],dx3[k+1],dmx3[k],dmx3[k+1]);

  const real_t conv_x = (U[c] * fx.ded_lo + U[xp] * fx.ded_hi) * real_t{0.5};
  const real_t conv_y = (V[c] * fy.ded_lo + V[yp] * fy.ded_hi) * real_t{0.5};
  const real_t conv_z = (W[c] * fz.ded_lo + W[zp] * fz.ded_hi) * real_t{0.5};
  real_t R = -real_t{0.5} * (conv_x + conv_y + conv_z);

  const real_t ve_x = (fx.kPP_hi*fx.ded_hi - fx.kPP_lo*fx.ded_lo
                     - fx.e_hi*fx.dkPP_hi*fx.ded_hi + fx.e_lo*fx.dkPP_lo*fx.ded_lo) / dx1[i];
  const real_t ve_y = (fy.kPP_hi*fy.ded_hi - fy.kPP_lo*fy.ded_lo
                     - fy.e_hi*fy.dkPP_hi*fy.ded_hi + fy.e_lo*fy.dkPP_lo*fy.ded_lo) / dx2[j];
  const real_t ve_z = (fz.kPP_hi*fz.ded_hi - fz.kPP_lo*fz.ded_lo
                     - fz.e_hi*fz.dkPP_hi*fz.ded_hi + fz.e_lo*fz.dkPP_lo*fz.ded_lo) / dx3[k];

  const real_t div_sum = div_bare(fx,dx1[i]) + div_bare(fy,dx2[j]) + div_bare(fz,dx3[k]);
  R += irc_h * (ve_x + ve_y + ve_z) - dirc_h * T[c] * div_sum;
  return R;
}

} // namespace stencil
} // namespace mpmstd::equation::scalar::kernels
