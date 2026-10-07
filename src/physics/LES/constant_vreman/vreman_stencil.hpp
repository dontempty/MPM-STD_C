#pragma once
// Const-Vreman (Vreman 2004) 셀 단위 식 — 참조 Fortran MPM-STD core_LES.f90
// LES_const_vreman_cuda 1:1 (감쇠함수 damping ≡ 1, IBM 마스크 H 는 미사용).
//   속도구배 a_mn = ∂u_n/∂x_m 은 공통 헤더 physics/LES/velocity_gradient.hpp (WALE 와 공유).
//
//   β_mn = Σ_l Δ_l² α_lm α_ln  (Δ_l = 그 방향 셀폭; 격자 비등방 내장)
//   B_β  = β11β22 − β12² + β11β33 − β13² + β22β33 − β23²
//   ν_t  = C_v √(B_β / α_mnα_mn),  α_mnα_mn < 1e-14 → 0,  C_v = 2.5 C_s²
#include <cmath>

#include "common/hd.hpp"
#include "common/types.hpp"
#include "physics/LES/velocity_gradient.hpp"

namespace mpmstd::physics::les::stencil {

// 셀 (i,j,k) 의 ν_t
MPMSTD_HD inline real_t eddy_viscosity_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* U, const real_t* V, const real_t* W, const real_t* dx_x, const real_t* dmx_x,
    const real_t* dx_y, const real_t* dmx_y, const real_t* dx_z, const real_t* dmx_z,
    const real_t c_vreman) {
  const VelocityGradient a = velocity_gradient_cell(i, j, k, ny_total, nz_total, U, V, W, dx_x,
                                                    dmx_x, dx_y, dmx_y, dx_z, dmx_z);
  const real_t a11 = a.a11, a12 = a.a12, a13 = a.a13;
  const real_t a21 = a.a21, a22 = a.a22, a23 = a.a23;
  const real_t a31 = a.a31, a32 = a.a32, a33 = a.a33;
  const real_t d1 = dx_x[i] * dx_x[i], d2 = dx_y[j] * dx_y[j], d3 = dx_z[k] * dx_z[k];
  // β_mn = Σ_l Δ_l² α_lm α_ln   (Fortran BBmn: DVmDX1·DVnDX1·dx1² + DVmDX2·DVnDX2·dx2² + DVmDX3·DVnDX3·dx3²)
  // 여기서 Fortran DVmDXl = ∂u_m/∂x_l = a_lm
  const real_t b11 = a11 * a11 * d1 + a21 * a21 * d2 + a31 * a31 * d3;
  const real_t b22 = a12 * a12 * d1 + a22 * a22 * d2 + a32 * a32 * d3;
  const real_t b33 = a13 * a13 * d1 + a23 * a23 * d2 + a33 * a33 * d3;
  const real_t b12 = a11 * a12 * d1 + a21 * a22 * d2 + a31 * a32 * d3;
  const real_t b13 = a11 * a13 * d1 + a21 * a23 * d2 + a31 * a33 * d3;
  const real_t b23 = a12 * a13 * d1 + a22 * a23 * d2 + a32 * a33 * d3;
  const real_t bb = b11 * b22 - b12 * b12 + b11 * b33 - b13 * b13 + b22 * b33 - b23 * b23;
  const real_t aa = a11 * a11 + a12 * a12 + a13 * a13 + a21 * a21 + a22 * a22 + a23 * a23
                    + a31 * a31 + a32 * a32 + a33 * a33;
  real_t ratio = bb / aa;
  if (std::abs(aa) < real_t(1e-14)) ratio = 0;
  return c_vreman * std::sqrt(std::abs(ratio));
}

}  // namespace mpmstd::physics::les::stencil
