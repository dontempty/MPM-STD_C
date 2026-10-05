#pragma once
// 가변-κ̂ (NOB) 에너지 방정식의 셀 단위 수식 — PaScaL_TCS module_solve_thermal
// (델타형 3-스윕 ADI) 를 MPM-STD_C 를 거쳐 1:1 이식. CPU 루프(system_variable.cpp)
// 와 GPU 커널(system_variable.cu)이 이 한 곳의 식을 같이 쓴다 (D3).
//
//   ∂θ/∂t = −u·∇θ + Ct/(ρ̂Ĉp)·∇·(κ̂∇θ),   Ct = 1/√(Ra·Pr)
//
// 한 축의 기여는 variable_axis_band() 가 전부 만든다:
//   * 밴드 eAM/eAC/eAP = [D] CN½ 가변-κ̂ 확산 + [L] dκ̂ 선형화 + [C] ¼ 준음해
//     이류 (상수물성 OB 모듈과 달리 대류도 절반이 음해로 들어간다 — 참조 스킴)
//   * want_rhs(첫 스윕)일 때만 R 재료: div_axis(∇·κ̂∇θ), convect_axis(이류),
//     viscous_axis(확산 − θ̂dκ̂ 선형화 몫)
// build_variable_cell() 이 스윕 단계에 따라 이를 호출한다:
//   첫 스윕  : 세 축 모두 → D = dt·(R − ΣM·θⁿ + d(1/ρ̂Ĉp)·θ·Σdiv), 대각 보정 포함
//   나중 스윕: 스윕 축 밴드만, D = 이전 스윕의 증분 (전치 통과)
//
// 벽(z) 마스크 2종 — 참조의 jem/jep 닫기를 그대로 재현한다:
//   wall_lo/hi      : 벽 면의 dκ̂ 를 0 으로 (밴드에서만; R 는 마스크 전 값 사용)
//   full_offdiag    : 나중 스윕의 벽 행 비대각 전체 제거 (δθ_wall = 0 닫기 —
//                     OB 모듈의 fold_z_wall_mirror 대신 참조는 이 마스크가 담당)

#include "common/hd.hpp"
#include "common/types.hpp"
#include "equation/property_access.hpp"

namespace mpmstd::equation::scalar::stencil {

MPMSTD_HD inline long long variable_index(const int i, const int j, const int k,
                                          const int ny_total, const int nz_total) {
  return (static_cast<long long>(i) * ny_total + j) * nz_total + k;
}

// ── 한 축의 밴드 + R 기여 (참조 scalar_var_stencil.hpp axis_band 1:1) ─────────
template <class KappaArray, class DKappaArray>
MPMSTD_HD inline void variable_axis_band(
    const real_t* T, const KappaArray conductivity,
    const DKappaArray conductivity_derivative, const real_t* face_velocity, const long long cm, const long long cc,
    const long long cp, const real_t dx_m, const real_t dx_c, const real_t dx_p,
    const real_t dmx_lo, const real_t dmx_hi, const real_t half_ct_inverse_heat_capacity,
    const bool wall_lo, const bool wall_hi, const bool full_offdiag_mask,
    const bool want_rhs, real_t& eAM, real_t& eAC, real_t& eAP, real_t& div_axis,
    real_t& convect_axis, real_t& viscous_axis) {
  const real_t half = real_t(0.5), quarter = real_t(0.25);

  // 면 값 (체적가중 보간) — R 과 밴드가 공유
  const real_t ded_lo = (T[cc] - T[cm]) / dmx_lo;
  const real_t ded_hi = (T[cp] - T[cc]) / dmx_hi;
  const real_t kappa_lo = (half / dmx_lo) * (dx_c * conductivity[cm] + dx_m * conductivity[cc]);
  const real_t kappa_hi = (half / dmx_hi) * (dx_p * conductivity[cc] + dx_c * conductivity[cp]);
  real_t dkappa_lo =
      (half / dmx_lo) * (dx_c * conductivity_derivative[cm] + dx_m * conductivity_derivative[cc]);
  real_t dkappa_hi =
      (half / dmx_hi) * (dx_p * conductivity_derivative[cc] + dx_c * conductivity_derivative[cp]);

  const real_t vlo = face_velocity[cc], vhi = face_velocity[cp];

  div_axis = (kappa_hi * ded_hi - kappa_lo * ded_lo) / dx_c;

  // R 전용 (첫 스윕) — 마스크 전 dκ̂ 로 계산 (참조 ORDER 그대로)
  if (want_rhs) {
    const real_t theta_lo = (dx_c * T[cm] + dx_m * T[cc]) * (half / dmx_lo);
    const real_t theta_hi = (dx_p * T[cc] + dx_c * T[cp]) * (half / dmx_hi);
    convect_axis = (vlo * ded_lo + vhi * ded_hi) * half;
    viscous_axis = (kappa_hi * ded_hi - kappa_lo * ded_lo - theta_hi * dkappa_hi * ded_hi
                    + theta_lo * dkappa_lo * ded_lo)
                   / dx_c;
  }

  // 음해 연산자: 벽 면의 dκ̂ 마스크
  if (wall_lo) dkappa_lo = 0;
  if (wall_hi) dkappa_hi = 0;

  const real_t cf = half_ct_inverse_heat_capacity / dx_c;
  eAC = +cf * kappa_hi / dmx_hi + cf * kappa_lo / dmx_lo
        - cf * dx_p * (half / dmx_hi) * dkappa_hi * ded_hi
        + cf * dx_m * (half / dmx_lo) * dkappa_lo * ded_lo + quarter * vlo / dmx_lo
        - quarter * vhi / dmx_hi;
  eAP = -cf * kappa_hi / dmx_hi - cf * dx_c * (half / dmx_hi) * dkappa_hi * ded_hi
        + quarter * vhi / dmx_hi;
  eAM = -cf * kappa_lo / dmx_lo + cf * dx_c * (half / dmx_lo) * dkappa_lo * ded_lo
        - quarter * vlo / dmx_lo;

  if (full_offdiag_mask) {
    if (wall_lo) eAM = 0;
    if (wall_hi) eAP = 0;
  }
}

// ── 셀 하나의 밴드(am, ac, ap) + 팩된 D (참조 build_A_rhs_cell 1:1) ──────────
// sweep_axis: 0=x, 1=y, 2=z (이 커널은 참조 1:1 이식이라 축 인자를 유지한다 —
// 호출자는 system_variable 의 build_x/y/z_system_variable 세 함수로 명시 분기).
template <class KappaArray, class DKappaArray, class HeatCapArray, class DHeatCapArray>
MPMSTD_HD inline void build_variable_cell(
    const int sweep_axis, const int i, const int j, const int k, const int ny_total,
    const int nz_total, const real_t* source, const real_t* T, const real_t* U,
    const real_t* V, const real_t* W, const KappaArray conductivity,
    const DKappaArray conductivity_derivative, const HeatCapArray inverse_heat_capacity,
    const DHeatCapArray inverse_heat_capacity_derivative, const real_t* dx_x,
    const real_t* dmx_x, const real_t* dx_y, const real_t* dmx_y, const real_t* dx_z,
    const real_t* dmx_z, const real_t Ct, const real_t dt, const bool is_first_sweep,
    const bool z_wall_lo, const bool z_wall_hi, real_t& am, real_t& ac, real_t& ap,
    real_t& D_value) {
  const long long c = variable_index(i, j, k, ny_total, nz_total);
  const real_t half_ct_ihc = real_t(0.5) * Ct * inverse_heat_capacity[c];
  const real_t half_ct_dihc = real_t(0.5) * Ct * inverse_heat_capacity_derivative[c];

  real_t eAM_d, eAC_d, eAP_d;

  if (is_first_sweep) {
    const long long xm = variable_index(i - 1, j, k, ny_total, nz_total);
    const long long xp = variable_index(i + 1, j, k, ny_total, nz_total);
    const long long ym = variable_index(i, j - 1, k, ny_total, nz_total);
    const long long yp = variable_index(i, j + 1, k, ny_total, nz_total);
    const long long zm = variable_index(i, j, k - 1, ny_total, nz_total);
    const long long zp = variable_index(i, j, k + 1, ny_total, nz_total);

    real_t eAMx, eACx, eAPx, dvx, cvx, vsx;
    real_t eAMy, eACy, eAPy, dvy, cvy, vsy;
    real_t eAMz, eACz, eAPz, dvz, cvz, vsz;
    variable_axis_band(T, conductivity, conductivity_derivative, U, xm, c, xp,
                       dx_x[i - 1], dx_x[i], dx_x[i + 1], dmx_x[i], dmx_x[i + 1],
                       half_ct_ihc, false, false, false, true, eAMx, eACx, eAPx, dvx,
                       cvx, vsx);
    variable_axis_band(T, conductivity, conductivity_derivative, V, ym, c, yp,
                       dx_y[j - 1], dx_y[j], dx_y[j + 1], dmx_y[j], dmx_y[j + 1],
                       half_ct_ihc, false, false, false, true, eAMy, eACy, eAPy, dvy,
                       cvy, vsy);
    variable_axis_band(T, conductivity, conductivity_derivative, W, zm, c, zp,
                       dx_z[k - 1], dx_z[k], dx_z[k + 1], dmx_z[k], dmx_z[k + 1],
                       half_ct_ihc, z_wall_lo, z_wall_hi, /*full_offdiag=*/false, true,
                       eAMz, eACz, eAPz, dvz, cvz, vsz);

    const real_t div_sum = dvx + dvy + dvz;
    const real_t R = -real_t(0.5) * (cvx + cvy + cvz) + half_ct_ihc * (vsx + vsy + vsz)
                     - half_ct_dihc * T[c] * div_sum;

    if (sweep_axis == 0) { eAM_d = eAMx; eAC_d = eACx; eAP_d = eAPx; }
    else if (sweep_axis == 1) { eAM_d = eAMy; eAC_d = eACy; eAP_d = eAPy; }
    else { eAM_d = eAMz; eAC_d = eACz; eAP_d = eAPz; }

    ac = eAC_d * dt - half_ct_dihc * div_sum * dt + real_t(1);
    ap = eAP_d * dt;
    am = eAM_d * dt;
    const real_t MxTn = eAPx * T[xp] + eACx * T[c] + eAMx * T[xm];
    const real_t MyTn = eAPy * T[yp] + eACy * T[c] + eAMy * T[ym];
    const real_t MzTn = eAPz * T[zp] + eACz * T[c] + eAMz * T[zm];
    D_value = (R - (MxTn + MyTn + MzTn) + half_ct_dihc * T[c] * div_sum) * dt;
  } else {
    real_t dv, cv, vs;  // want_rhs = false — 채워지지 않는 출력
    if (sweep_axis == 0) {
      const long long xm = variable_index(i - 1, j, k, ny_total, nz_total);
      const long long xp = variable_index(i + 1, j, k, ny_total, nz_total);
      variable_axis_band(T, conductivity, conductivity_derivative, U, xm, c, xp,
                         dx_x[i - 1], dx_x[i], dx_x[i + 1], dmx_x[i], dmx_x[i + 1],
                         half_ct_ihc, false, false, false, false, eAM_d, eAC_d, eAP_d,
                         dv, cv, vs);
    } else if (sweep_axis == 1) {
      const long long ym = variable_index(i, j - 1, k, ny_total, nz_total);
      const long long yp = variable_index(i, j + 1, k, ny_total, nz_total);
      variable_axis_band(T, conductivity, conductivity_derivative, V, ym, c, yp,
                         dx_y[j - 1], dx_y[j], dx_y[j + 1], dmx_y[j], dmx_y[j + 1],
                         half_ct_ihc, false, false, false, false, eAM_d, eAC_d, eAP_d,
                         dv, cv, vs);
    } else {
      const long long zm = variable_index(i, j, k - 1, ny_total, nz_total);
      const long long zp = variable_index(i, j, k + 1, ny_total, nz_total);
      variable_axis_band(T, conductivity, conductivity_derivative, W, zm, c, zp,
                         dx_z[k - 1], dx_z[k], dx_z[k + 1], dmx_z[k], dmx_z[k + 1],
                         half_ct_ihc, z_wall_lo, z_wall_hi, /*full_offdiag=*/true,
                         false, eAM_d, eAC_d, eAP_d, dv, cv, vs);
    }
    ac = eAC_d * dt + real_t(1);
    ap = eAP_d * dt;
    am = eAM_d * dt;
    D_value = source[c];  // 이전 스윕의 전치 증분 통과
  }
}

}  // namespace mpmstd::equation::scalar::stencil
