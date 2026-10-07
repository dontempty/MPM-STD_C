#pragma once
// WALE 와점성 (Nicoud & Ducros 1999, Flow Turbul. Combust. 62, 183–200) 셀 단위 식.
//   g_ij = ∂u_i/∂x_j  (velocity_gradient.hpp 의 a_mn = ∂u_n/∂x_m 의 전치: g_ij = a_ji)
//   S_ij   = ½(g_ij + g_ji)                       변형률
//   g²_ij  = g_ik g_kj
//   S^d_ij = ½(g²_ij + g²_ji) − ⅓ δ_ij g²_kk       속도구배 제곱의 무흔적 대칭부
//   ν_t = (C_w Δ)² · (S^d:S^d)^{3/2} / [ (S:S)^{5/2} + (S^d:S^d)^{5/4} ],   Δ = (Δx Δy Δz)^{1/3}
//
//   성질: 순수 전단(벽 근처 ∂u/∂z 만 있는 흐름)에서 S^d = 0 → ν_t ∝ y³ (감쇠함수·벽거리 불요,
//   IBM 호환). 분모가 0 이면(정지 유체) ν_t = 0.  C_w = 0.5 (논문 권장, C_w² ≈ 10.6 C_s²).
#include <cmath>

#include "common/hd.hpp"
#include "common/types.hpp"
#include "physics/LES/velocity_gradient.hpp"

namespace mpmstd::physics::les::stencil {

// 셀 (i,j,k) 의 ν_t
MPMSTD_HD inline real_t eddy_viscosity_cell_wale(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* U, const real_t* V, const real_t* W, const real_t* dx_x, const real_t* dmx_x,
    const real_t* dx_y, const real_t* dmx_y, const real_t* dx_z, const real_t* dmx_z,
    const real_t c_wale) {
  const VelocityGradient a = velocity_gradient_cell(i, j, k, ny_total, nz_total, U, V, W, dx_x,
                                                    dmx_x, dx_y, dmx_y, dx_z, dmx_z);
  // g_ij = ∂u_i/∂x_j = a_ji
  const real_t g11 = a.a11, g12 = a.a21, g13 = a.a31;
  const real_t g21 = a.a12, g22 = a.a22, g23 = a.a32;
  const real_t g31 = a.a13, g32 = a.a23, g33 = a.a33;
  // g² = g·g
  const real_t q11 = g11 * g11 + g12 * g21 + g13 * g31;
  const real_t q12 = g11 * g12 + g12 * g22 + g13 * g32;
  const real_t q13 = g11 * g13 + g12 * g23 + g13 * g33;
  const real_t q21 = g21 * g11 + g22 * g21 + g23 * g31;
  const real_t q22 = g21 * g12 + g22 * g22 + g23 * g32;
  const real_t q23 = g21 * g13 + g22 * g23 + g23 * g33;
  const real_t q31 = g31 * g11 + g32 * g21 + g33 * g31;
  const real_t q32 = g31 * g12 + g32 * g22 + g33 * g32;
  const real_t q33 = g31 * g13 + g32 * g23 + g33 * g33;
  const real_t trace_third = (q11 + q22 + q33) / real_t(3);
  // S^d (대칭, 무흔적)
  const real_t sd11 = q11 - trace_third, sd22 = q22 - trace_third, sd33 = q33 - trace_third;
  const real_t sd12 = real_t(0.5) * (q12 + q21);
  const real_t sd13 = real_t(0.5) * (q13 + q31);
  const real_t sd23 = real_t(0.5) * (q23 + q32);
  // S (대칭)
  const real_t s11 = g11, s22 = g22, s33 = g33;
  const real_t s12 = real_t(0.5) * (g12 + g21);
  const real_t s13 = real_t(0.5) * (g13 + g31);
  const real_t s23 = real_t(0.5) * (g23 + g32);
  const real_t sdsd = sd11 * sd11 + sd22 * sd22 + sd33 * sd33
                      + real_t(2) * (sd12 * sd12 + sd13 * sd13 + sd23 * sd23);
  const real_t ss = s11 * s11 + s22 * s22 + s33 * s33
                    + real_t(2) * (s12 * s12 + s13 * s13 + s23 * s23);
  // (S:S)^{5/2} + (S^d:S^d)^{5/4}  — pow 대신 sqrt 조합 (양수라 안전)
  const real_t ss_sqrt = std::sqrt(ss), sdsd_sqrt = std::sqrt(sdsd);
  const real_t denominator = ss * ss * ss_sqrt + sdsd * std::sqrt(sdsd_sqrt);
  if (!(denominator > real_t(0))) return 0;  // 정지 유체 (S = S^d = 0)
  const real_t delta = std::cbrt(dx_x[i] * dx_y[j] * dx_z[k]);
  return c_wale * c_wale * delta * delta * (sdsd * sdsd_sqrt) / denominator;
}

}  // namespace mpmstd::physics::les::stencil
