#pragma once

// MPMSTD_HD per-cell stencil for the variable-κ̂ (NOB) scalar scheme.
// Included by both scalar_var_kernels_cpu.cpp and scalar_gpu.cu.
//
// PaScaL_TCS module_solve_thermal.f90 (mpi_thermal_solver)의 스킴을 이식.
// 한 축의 계산은 axis_band()로 모으고(공식 단일 소스), build_A_rhs_cell()이
// stage에 따라 이를 호출한다:
//   stage 1 : x·y·z 3축 모두 필요(RHS 팩 = ΣM·θⁿ + R) → 3번 호출
//   stage 2/3 : sweep 축 밴드만 필요 → sweep 축 1번만 호출 (안 쓰는 축 계산·
//               로드 없음). R 전용 계산(면 T·대류·확산)은 want_rhs로 건너뜀.

#include "common/macros.hpp"
#include "common/types.hpp"
#include "common/direction.hpp"

namespace mpmstd::equation::scalar::kernels {
namespace stencil {

static constexpr int kH = 1;  // halo width

MPMSTD_HD int idx3(int i, int j, int k, int n2, int n3) {
  return (i * n2 + j) * n3 + k;
}

// ── 한 축(sweep 방향 무관, 임의의 한 방향)의 밴드 + R 기여 ────────────────────
// 항상 채움: eAM, eAC, eAP  (그 축의 음해 삼중대각 밴드)
// want_rhs일 때만 채움: div_axis, convect_axis, viscous_axis  (stage-1 R/RHS 팩용)
//
// 인자: 이 축의 이웃 플랫 인덱스 cm/cc/cp, 면 속도는 vel[cc]·vel[cp](=lo·hi 면),
//   격자 dx_{m,c,p}·dmx_{lo,hi}, 계수 invRhoCpCt_half.
// wall_lo/wall_hi: jem/jep dkPP 마스크(벽 축에서만 참). full_offdiag_mask: 벽에서
//   off-diagonal 전체를 0으로(=stages 2/3). ORDER: viscous(R)는 마스크 전 dkPP로,
//   밴드는 마스크 후 dkPP로 계산 — TCS와 동일.
MPMSTD_HD void
axis_band(const real_t* T, const real_t* kappa, const real_t* dkappa, const real_t* vel,
          int cm, int cc, int cp,
          real_t dx_m, real_t dx_c, real_t dx_p, real_t dmx_lo, real_t dmx_hi,
          real_t invRhoCpCt_half,
          bool wall_lo, bool wall_hi, bool full_offdiag_mask, bool want_rhs,
          real_t& eAM, real_t& eAC, real_t& eAP,
          real_t& div_axis, real_t& convect_axis, real_t& viscous_axis) {
  const real_t half{0.5}, quarter{0.25};

  // 면 값 (체적가중 보간, TCS ref 257-289) — R과 밴드가 공유
  const real_t ded_lo = (T[cc] - T[cm]) / dmx_lo;              // 면 온도구배  (TCS dedx1/2)
  const real_t ded_hi = (T[cp] - T[cc]) / dmx_hi;
  const real_t kPP_lo = (half/dmx_lo) * (dx_c*kappa[cm] + dx_m*kappa[cc]);   // 면 κ̂ (TCS kPP1/2)
  const real_t kPP_hi = (half/dmx_hi) * (dx_p*kappa[cc] + dx_c*kappa[cp]);
  real_t       dkPP_lo = (half/dmx_lo) * (dx_c*dkappa[cm] + dx_m*dkappa[cc]); // 면 dκ̂ (TCS dkPP1/2)
  real_t       dkPP_hi = (half/dmx_hi) * (dx_p*dkappa[cc] + dx_c*dkappa[cp]);

  const real_t vlo = vel[cc], vhi = vel[cp];   // lo·hi 면 법선속도

  div_axis = (kPP_hi*ded_hi - kPP_lo*ded_lo) / dx_c;   // ∇·(κ̂∇θ) 이 축 성분(선형화 없이)

  // ── R 전용(stage-1) — 마스크 전 unmasked dkPP로 계산 ────────────────────────
  if (want_rhs) {
    // 면 보간 온도 θ̂ (viscous 선형화에만 쓰임; 밴드엔 불필요 → stage-1에서만 계산)
    const real_t e_lo = (dx_c*T[cm] + dx_m*T[cc]) * (half/dmx_lo);
    const real_t e_hi = (dx_p*T[cc] + dx_c*T[cp]) * (half/dmx_hi);
    // ① 대류 (이류형): 셀 양쪽 면 (속도×구배) 0.5 평균 (TCS convect_e)
    convect_axis = (vlo*ded_lo + vhi*ded_hi) * half;
    // ② 변수-κ̂ 확산: [κ̂·∂θ] 발산 − [θ̂·dκ̂·∂θ] 선형화 몫 (TCS viscous_e)
    viscous_axis = (kPP_hi*ded_hi - kPP_lo*ded_lo
                  - e_hi*dkPP_hi*ded_hi + e_lo*dkPP_lo*ded_lo) / dx_c;
  }

  // ── 음해 연산자: jem/jep dkPP 마스크 적용(벽 면에서만) ──────────────────────
  if (wall_lo) dkPP_lo = real_t{0};
  if (wall_hi) dkPP_hi = real_t{0};

  // 밴드 (부호를 항 앞으로; [D] 확산 · [L] dκ̂ 선형화 · [C] 준음해 대류)
  // (M·δθ)_c = eAM·δθ_m + eAC·δθ_c + eAP·δθ_p.  TCS eACI/eAPI/eAMI (ref 352-357).
  const real_t cf = invRhoCpCt_half / dx_c;
  eAC = + cf*kPP_hi/dmx_hi                     // [D] hi면 유출(대각)
        + cf*kPP_lo/dmx_lo                      // [D] lo면 유출(대각)
        - cf*dx_p*(half/dmx_hi)*dkPP_hi*ded_hi  // [L] hi면 δθ̂ 중 δθ_c 몫
        + cf*dx_m*(half/dmx_lo)*dkPP_lo*ded_lo  // [L] lo면 δθ̂ 중 δθ_c 몫
        + quarter*vlo/dmx_lo                    // [C] lo면 속도 × δθ_c
        - quarter*vhi/dmx_hi;                   // [C] hi면 속도 × δθ_c
  eAP = - cf*kPP_hi/dmx_hi                      // [D] hi면 → δθ_p 결합
        - cf*dx_c*(half/dmx_hi)*dkPP_hi*ded_hi  // [L] hi면 δθ̂ 중 δθ_p 몫
        + quarter*vhi/dmx_hi;                   // [C] hi면 속도 × δθ_p
  eAM = - cf*kPP_lo/dmx_lo                      // [D] lo면 → δθ_m 결합
        + cf*dx_c*(half/dmx_lo)*dkPP_lo*ded_lo  // [L] lo면 δθ̂ 중 δθ_m 몫
        - quarter*vlo/dmx_lo;                   // [C] lo면 속도 × δθ_m

  // 벽 셀 off-diagonal 전체 제거 (stages 2/3; stage-1은 M·θⁿ 팩에 필요해 유지)
  if (full_offdiag_mask) {
    if (wall_lo) eAM = real_t{0};
    if (wall_hi) eAP = real_t{0};
  }
}

// Per-cell build_A_RHS — the single implementation called by both the CPU loop
// (build_A_rhs_nob) and the GPU kernel (k_build_A_rhs_nob). Writes the sweep-
// axis (d) band (am, ac, ap) AND the packed RHS Dval at cell (i,j,k).
//
// 에너지 방정식  ∂θ/∂t = −u·∇θ + Ct/(ρ̂Ĉp)·∇·(κ̂∇θ)   (Ct = 1/√(Ra·Pr))
// Stage 1 (is_stage1=true): R = 우변을 θⁿ에서 명시 평가한 잔차 = −½·(u·∇θ)
//   + ½·Ct/(ρ̂Ĉp)·[변수-κ̂ 확산 − θ̂·dκ̂ 선형화] − ½·Ct·d(1/ρ̂Ĉp)·θ·∇·(κ̂∇θ).
//   Dval = dt·( R − ΣM·θⁿ + dinvRhoCpCt_half·θ·div_sum ),  ac = eAC·dt − …·dt + 1.
// Stage 2/3: sweep 축 밴드만; Dval = src (이전 sweep의 전치 증분), ac = eAC·dt + 1.
// z_wall_lo/z_wall_hi: 물리 z-벽 소유 랭크에서만 참(jem/jep 마스크).
MPMSTD_HD void
build_A_rhs_cell(Direction d,
                 int i, int j, int k,
                 int n1_tot, int n2_tot, int n3_tot,
                 const real_t* src, const real_t* T,
                 const real_t* U, const real_t* V, const real_t* W,
                 const real_t* KPP, const real_t* dKPP,
                 const real_t* invRhoCp, const real_t* dinvRhoCp,
                 const real_t* dx1, const real_t* dmx1,
                 const real_t* dx2, const real_t* dmx2,
                 const real_t* dx3, const real_t* dmx3,
                 real_t Ct, real_t dt, bool is_stage1,
                 bool z_wall_lo, bool z_wall_hi,
                 real_t& am, real_t& ac, real_t& ap, real_t& Dval) {
  const real_t half{0.5};
  const int c = idx3(i, j, k, n2_tot, n3_tot);

  // CN 절반이 미리 곱해진 계수 (TCS invRhoCpCt_half / dinvRhoCpCt_half)
  const real_t invRhoCpCt_half  = half * Ct * invRhoCp[c];
  const real_t dinvRhoCpCt_half = half * Ct * dinvRhoCp[c];

  real_t eAM_d, eAC_d, eAP_d;   // sweep 축 밴드 (양 stage 공통 출력)

  if (is_stage1) {
    // ── stage 1: 세 축 모두 필요 (RHS 팩 = R − ΣM·θⁿ + …) ──────────────────
    const int xm = idx3(i-1,j,k, n2_tot,n3_tot), xp = idx3(i+1,j,k, n2_tot,n3_tot);
    const int ym = idx3(i,j-1,k, n2_tot,n3_tot), yp = idx3(i,j+1,k, n2_tot,n3_tot);
    const int zm = idx3(i,j,k-1, n2_tot,n3_tot), zp = idx3(i,j,k+1, n2_tot,n3_tot);

    real_t eAMx,eACx,eAPx, dvx,cvx,vsx;
    real_t eAMy,eACy,eAPy, dvy,cvy,vsy;
    real_t eAMz,eACz,eAPz, dvz,cvz,vsz;
    axis_band(T,KPP,dKPP,U, xm,c,xp, dx1[i-1],dx1[i],dx1[i+1], dmx1[i],dmx1[i+1],
              invRhoCpCt_half, false,false,false, true, eAMx,eACx,eAPx, dvx,cvx,vsx);
    axis_band(T,KPP,dKPP,V, ym,c,yp, dx2[j-1],dx2[j],dx2[j+1], dmx2[j],dmx2[j+1],
              invRhoCpCt_half, false,false,false, true, eAMy,eACy,eAPy, dvy,cvy,vsy);
    axis_band(T,KPP,dKPP,W, zm,c,zp, dx3[k-1],dx3[k],dx3[k+1], dmx3[k],dmx3[k+1],
              invRhoCpCt_half, z_wall_lo,z_wall_hi,/*full_mask=*/false, true,
              eAMz,eACz,eAPz, dvz,cvz,vsz);

    const real_t div_sum = dvx + dvy + dvz;
    const real_t RHS = -half*(cvx + cvy + cvz)
                     + invRhoCpCt_half*(vsx + vsy + vsz)
                     - dinvRhoCpCt_half * T[c] * div_sum;

    if      (d == Direction::X) { eAM_d=eAMx; eAC_d=eACx; eAP_d=eAPx; }
    else if (d == Direction::Y) { eAM_d=eAMy; eAC_d=eACy; eAP_d=eAPy; }
    else                        { eAM_d=eAMz; eAC_d=eACz; eAP_d=eAPz; }

    ac = eAC_d * dt - dinvRhoCpCt_half * div_sum * dt + real_t{1};
    ap = eAP_d * dt;
    am = eAM_d * dt;
    const real_t MxTn = eAPx*T[xp] + eACx*T[c] + eAMx*T[xm];
    const real_t MyTn = eAPy*T[yp] + eACy*T[c] + eAMy*T[ym];
    const real_t MzTn = eAPz*T[zp] + eACz*T[c] + eAMz*T[zm];
    Dval = ( RHS - (MxTn + MyTn + MzTn) + dinvRhoCpCt_half * T[c] * div_sum ) * dt;
  } else {
    // ── stage 2/3: sweep 축 밴드만 (안 쓰는 축은 계산·로드하지 않음) ─────────
    real_t dv, cv, vs;   // R 전용 출력 — want_rhs=false라 채워지지 않음
    if (d == Direction::X) {
      const int xm = idx3(i-1,j,k, n2_tot,n3_tot), xp = idx3(i+1,j,k, n2_tot,n3_tot);
      axis_band(T,KPP,dKPP,U, xm,c,xp, dx1[i-1],dx1[i],dx1[i+1], dmx1[i],dmx1[i+1],
                invRhoCpCt_half, false,false,false, false, eAM_d,eAC_d,eAP_d, dv,cv,vs);
    } else if (d == Direction::Y) {
      const int ym = idx3(i,j-1,k, n2_tot,n3_tot), yp = idx3(i,j+1,k, n2_tot,n3_tot);
      axis_band(T,KPP,dKPP,V, ym,c,yp, dx2[j-1],dx2[j],dx2[j+1], dmx2[j],dmx2[j+1],
                invRhoCpCt_half, false,false,false, false, eAM_d,eAC_d,eAP_d, dv,cv,vs);
    } else {
      const int zm = idx3(i,j,k-1, n2_tot,n3_tot), zp = idx3(i,j,k+1, n2_tot,n3_tot);
      axis_band(T,KPP,dKPP,W, zm,c,zp, dx3[k-1],dx3[k],dx3[k+1], dmx3[k],dmx3[k+1],
                invRhoCpCt_half, z_wall_lo,z_wall_hi,/*full_mask=*/true, false,
                eAM_d,eAC_d,eAP_d, dv,cv,vs);
    }
    ac = eAC_d * dt + real_t{1};
    ap = eAP_d * dt;
    am = eAM_d * dt;
    Dval = src[c];   // 이전 sweep의 전치 증분 통과 (ref 376/444)
  }
}

} // namespace stencil
} // namespace mpmstd::equation::scalar::kernels
