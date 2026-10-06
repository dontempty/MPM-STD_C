#pragma once
// Const-Vreman (Vreman 2004) 셀 단위 식 — 참조 Fortran MPM-STD core_LES.f90
// LES_const_vreman_cuda 1:1 (감쇠함수 damping ≡ 1, IBM 마스크 H 는 미사용).
//   축 대응: Fortran (x1 주류, x2 벽수직, x3 스팬) → 우리 (x, z, y). 식이 세 방향에 대칭이라
//   우리 (i,j,k) = (x,y,z) 순서로 그대로 쓴다 (dx·dmx 는 각 축의 cell_width·center_distance).
//
//   α_mn = ∂u_n/∂x_m (셀중심):  대각 = 면 차분, 비대각 = own 축 체적가중 중심보간 → 스윕 축 면 보간 → 차분
//   β_mn = Σ_l Δ_l² α_lm α_ln  (Δ_l = 그 방향 셀폭; 격자 비등방 내장)
//   B_β  = β11β22 − β12² + β11β33 − β13² + β22β33 − β23²
//   ν_t  = C_v √(B_β / α_mnα_mn),  α_mnα_mn < 1e-14 → 0,  C_v = 2.5 C_s²
#include <cmath>

#include "common/hd.hpp"
#include "common/types.hpp"

namespace mpmstd::physics::les::stencil {

MPMSTD_HD inline long long idx3(const int i, const int j, const int k, const int ny_total,
                                const int nz_total) {
  return (static_cast<long long>(i) * ny_total + j) * nz_total + k;
}

// own 축 a 의 면 속도 f(셀 c 의 마이너스 면, 플러스 면 = c + stride_a) 를 셀중심으로:
//   ½/dx_a(c)·(dmx_a(c)·f(plus) + dmx_a(plus)·f(c))   — Fortran 의 체적가중 (균일이면 ½(f⁺+f⁻))
MPMSTD_HD inline real_t own_center(const real_t* f, const long long c, const long long stride_a,
                                   const real_t dx_c, const real_t dmx_c, const real_t dmx_p) {
  return real_t(0.5) / dx_c * (dmx_c * f[c + stride_a] + dmx_p * f[c]);
}

// 셀중심 보간된 f 를 스윕 축 b 의 마이너스 면(b−½)으로: (f_c·½dx_b(m) + f_m·½dx_b(c)) / dmx_b(c)
MPMSTD_HD inline real_t to_minus_face(const real_t f_c, const real_t f_m, const real_t dx_b_c,
                                      const real_t dx_b_m, const real_t dmx_b_c) {
  return (f_c * real_t(0.5) * dx_b_m + f_m * real_t(0.5) * dx_b_c) / dmx_b_c;
}

// 성분 a(면 속도 f, own 축 a) 의 축 b 방향 도함수 ∂f/∂x_b (셀중심), b ≠ a
MPMSTD_HD inline real_t cross_gradient(const real_t* f, const long long c, const long long stride_a,
                                       const long long stride_b, const int ia, const int ib,
                                       const real_t* dx_a, const real_t* dmx_a, const real_t* dx_b,
                                       const real_t* dmx_b) {
  // own 축 중심보간 (셀 c, c−b, c+b)
  const real_t fc = own_center(f, c, stride_a, dx_a[ia], dmx_a[ia], dmx_a[ia + 1]);
  const real_t fm = own_center(f, c - stride_b, stride_a, dx_a[ia], dmx_a[ia], dmx_a[ia + 1]);
  const real_t fp = own_center(f, c + stride_b, stride_a, dx_a[ia], dmx_a[ia], dmx_a[ia + 1]);
  const real_t f1 = to_minus_face(fc, fm, dx_b[ib], dx_b[ib - 1], dmx_b[ib]);
  const real_t f2 = to_minus_face(fp, fc, dx_b[ib + 1], dx_b[ib], dmx_b[ib + 1]);
  return (f2 - f1) / dx_b[ib];
}

// 셀 (i,j,k) 의 ν_t
MPMSTD_HD inline real_t eddy_viscosity_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* U, const real_t* V, const real_t* W, const real_t* dx_x, const real_t* dmx_x,
    const real_t* dx_y, const real_t* dmx_y, const real_t* dx_z, const real_t* dmx_z,
    const real_t c_vreman) {
  const long long sx = static_cast<long long>(ny_total) * nz_total, sy = nz_total, sz = 1;
  const long long c = idx3(i, j, k, ny_total, nz_total);
  // 대각: 면 차분
  const real_t a11 = (U[c + sx] - U[c]) / dx_x[i];  // ∂u/∂x
  const real_t a22 = (V[c + sy] - V[c]) / dx_y[j];  // ∂v/∂y
  const real_t a33 = (W[c + sz] - W[c]) / dx_z[k];  // ∂w/∂z
  // 비대각 (Fortran DV1DX2 = ∂u/∂x2 등): a_mn = ∂u_n/∂x_m
  const real_t a21 = cross_gradient(U, c, sx, sy, i, j, dx_x, dmx_x, dx_y, dmx_y);  // ∂u/∂y
  const real_t a31 = cross_gradient(U, c, sx, sz, i, k, dx_x, dmx_x, dx_z, dmx_z);  // ∂u/∂z
  const real_t a12 = cross_gradient(V, c, sy, sx, j, i, dx_y, dmx_y, dx_x, dmx_x);  // ∂v/∂x
  const real_t a32 = cross_gradient(V, c, sy, sz, j, k, dx_y, dmx_y, dx_z, dmx_z);  // ∂v/∂z
  const real_t a13 = cross_gradient(W, c, sz, sx, k, i, dx_z, dmx_z, dx_x, dmx_x);  // ∂w/∂x
  const real_t a23 = cross_gradient(W, c, sz, sy, k, j, dx_z, dmx_z, dx_y, dmx_y);  // ∂w/∂y
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
