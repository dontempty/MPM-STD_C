#pragma once
// 가변 점성(NOB) 운동량의 셀 단위 수식 — PaScaL_TCS module_solve_momentum.f90
// (solvedU/V/W + blockLdV/LdU) 를 MPM-STD_C momentum_var_kernels 를 거쳐 1:1
// 이식. CPU 루프(rhs.cpp 등)와 GPU 커널(.cu)이 이 한 곳을 같이 쓴다.
//
// 축 맞바꿈 (참조 → 우리):  x1(U, 주류, 주기) → x ;  x2(V, 벽수직) → z(W) ;
//                           x3(W, 스팬, 주기) → y(V).  벽은 z 에만 있다.
// 성분별 구조 (own = 자기 방향):
//   * own 방향 확산은 ×2 (셀 μ — own-x 는 참조의 μ 비대칭까지 재현: μ≡1 소거),
//     cross 방향은 ×1 에 **이중 체적가중 면 μ** (own 축으로 먼저, 스윕 축으로
//     다시 보간 — μ≡1 이면 항등).
//   * 대류는 ¼ 준음해 + own 방향의 ½·∂q 자기미분.
//   * cross-stress(전치 응력)·cross 대류 북키핑은 R 에서 빼 둔다.
//   * 압력·부력은 여기 없다 — main 이 += 로 조립 (D-규칙).
// 벽 닫기는 OB 와 동일: zero-ghost + fold/핀 (jum/jup 마스크 없음).
// 증분 커플링(M21·M31·M32 전진 / M23·M12·M13 사후)은 별도 셀 함수.

#include "common/hd.hpp"
#include "common/types.hpp"
#include "equation/property_access.hpp"  // UnitArray — 상수물성 경로가 물성 배열 자리에 끼운다

namespace mpmstd::equation::momentum::stencil {

MPMSTD_HD inline long long vindex(const int i, const int j, const int k,
                                  const int ny_total, const int nz_total) {
  return (static_cast<long long>(i) * ny_total + j) * nz_total + k;
}

// 체적가중 2점 면 보간: 0.5·(dxA·fA + dxB·fB)/dmx (참조 mface)
MPMSTD_HD inline real_t face_blend(const real_t dxA, const real_t fA, const real_t dxB,
                                   const real_t fB, const real_t dmx) {
  return real_t(0.5) * (dxA * fA + dxB * fB) / dmx;
}

// ── ① stage-1 델타형 잔차  R·dt (성분별 — 참조 compute_mpmstd_rhs_var) ──────
// 반환값을 rhs 에 += 한다 (호출 전 rhs 는 0 또는 다른 명시항).

template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t rhs_u_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* U, const real_t* V, const real_t* W, const MuArray mu,
    const RhoArray invrho, const real_t* dx1, const real_t* dmx1, const real_t* dx2,
    const real_t* dmx2, const real_t* dx3, const real_t* dmx3, const real_t Cmu,
    const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25), kTwo = real_t(2);
  const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cim = vindex(im, j, k, ny_total, nz_total);
  const long long cip = vindex(ip, j, k, ny_total, nz_total);
  const long long cjm = vindex(i, jm, k, ny_total, nz_total);
  const long long cjp = vindex(i, jp, k, ny_total, nz_total);
  const long long ckm = vindex(i, j, km, ny_total, nz_total);
  const long long ckp = vindex(i, j, kp, ny_total, nz_total);

  const real_t u1 = kHalf * (U[cim] + U[c]);
  const real_t u2 = kHalf * (U[c] + U[cip]);
  const real_t w5 = face_blend(dx1[im], W[c], dx1[i], W[cim], dmx1[i]);
  const real_t w6 = face_blend(dx1[im], W[ckp], dx1[i], W[vindex(im, j, kp, ny_total, nz_total)], dmx1[i]);
  const real_t v3 = face_blend(dx1[im], V[c], dx1[i], V[cim], dmx1[i]);
  const real_t v4 = face_blend(dx1[im], V[cjp], dx1[i], V[vindex(im, jp, k, ny_total, nz_total)], dmx1[i]);

  const real_t dudx1 = (U[c] - U[cim]) / dx1[im];
  const real_t dudx2 = (U[cip] - U[c]) / dx1[i];
  const real_t dudz5 = (U[c] - U[ckm]) / dmx3[k];
  const real_t dudz6 = (U[ckp] - U[c]) / dmx3[kp];
  const real_t dudy3 = (U[c] - U[cjm]) / dmx2[j];
  const real_t dudy4 = (U[cjp] - U[c]) / dmx2[jp];
  const real_t dwdx5 = (W[c] - W[cim]) / dmx1[i];
  const real_t dwdx6 = (W[ckp] - W[vindex(im, j, kp, ny_total, nz_total)]) / dmx1[i];
  const real_t dvdx3 = (V[c] - V[cim]) / dmx1[i];
  const real_t dvdx4 = (V[cjp] - V[vindex(im, jp, k, ny_total, nz_total)]) / dmx1[i];

  const real_t za = face_blend(dx1[im], mu[ckm], dx1[i], mu[vindex(im, j, km, ny_total, nz_total)], dmx1[i]);
  const real_t zc = face_blend(dx1[im], mu[c], dx1[i], mu[cim], dmx1[i]);
  const real_t zb = face_blend(dx1[im], mu[ckp], dx1[i], mu[vindex(im, j, kp, ny_total, nz_total)], dmx1[i]);
  const real_t muz5 = kHalf * (dx3[km] * zc + dx3[k] * za) / dmx3[k];
  const real_t muz6 = kHalf * (dx3[kp] * zc + dx3[k] * zb) / dmx3[kp];
  const real_t ya = face_blend(dx1[im], mu[cjm], dx1[i], mu[vindex(im, jm, k, ny_total, nz_total)], dmx1[i]);
  const real_t yb = face_blend(dx1[im], mu[cjp], dx1[i], mu[vindex(im, jp, k, ny_total, nz_total)], dmx1[i]);
  const real_t muy3 = kHalf * (dx2[jm] * zc + dx2[j] * ya) / dmx2[j];
  const real_t muy4 = kHalf * (dx2[jp] * zc + dx2[j] * yb) / dmx2[jp];

  const real_t irc = face_blend(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]);
  const real_t inh = kHalf * Cmu * irc;

  const real_t vu1 = (mu[c] * dudx2 - mu[cim] * dudx1) / dmx1[i];
  const real_t vuZ = (muz6 * dudz6 - muz5 * dudz5) / dx3[k];
  const real_t vuY = (muy4 * dudy4 - muy3 * dudy3) / dx2[j];
  const real_t vuZc = (muz6 * dwdx6 - muz5 * dwdx5) / dx3[k];
  const real_t vuYc = (muy4 * dvdx4 - muy3 * dvdx3) / dx2[j];
  real_t R = inh * (kTwo * vu1 + vuZ + vuY + vuZc + vuYc);

  const real_t mACI = inh / dmx1[i] * (mu[c] / dx1[i] + mu[cim] / dx1[im]) * kTwo
                      + kQ * (kHalf * dudx2 + kHalf * dudx1 - u2 / dx1[i] + u1 / dx1[im]);
  const real_t mAPI = -inh / dmx1[i] * mu[c] / dx1[i] * kTwo + kQ * (u2 / dx1[i] + kHalf * dudx2);
  const real_t mAMI = -inh / dmx1[i] * mu[cim] / dx1[im] * kTwo + kQ * (-u1 / dx1[im] + kHalf * dudx1);
  const real_t mACK = inh / dx3[k] * (muz6 / dmx3[kp] + muz5 / dmx3[k]) + kQ * (-w6 / dmx3[kp] + w5 / dmx3[k]);
  const real_t mAPK = -inh / dx3[k] * muz6 / dmx3[kp] + kQ * (w6 / dmx3[kp]);
  const real_t mAMK = -inh / dx3[k] * muz5 / dmx3[k] + kQ * (-w5 / dmx3[k]);
  const real_t mACJ = inh / dx2[j] * (muy4 / dmx2[jp] + muy3 / dmx2[j]) + kQ * (-v4 / dmx2[jp] + v3 / dmx2[j]);
  const real_t mAPJ = -inh / dx2[j] * muy4 / dmx2[jp] + kQ * (v4 / dmx2[jp]);
  const real_t mAMJ = -inh / dx2[j] * muy3 / dmx2[j] + kQ * (-v3 / dmx2[j]);

  R -= (mAPI * U[cip] + mACI * U[c] + mAMI * U[cim] + mAPJ * U[cjp] + mACJ * U[c]
        + mAMJ * U[cjm] + mAPK * U[ckp] + mACK * U[c] + mAMK * U[ckm]);
  R -= (kQ * (w6 * dudz6 + w5 * dudz5) - inh / dx3[k] * (muz6 * dwdx6 - muz5 * dwdx5));
  R -= (kQ * (v4 * dudy4 + v3 * dudy3) - inh / dx2[j] * (muy4 * dvdx4 - muy3 * dvdx3));
  return R * dt;
}

template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t rhs_v_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* U, const real_t* V, const real_t* W, const MuArray mu,
    const RhoArray invrho, const real_t* dx1, const real_t* dmx1, const real_t* dx2,
    const real_t* dmx2, const real_t* dx3, const real_t* dmx3, const real_t Cmu,
    const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25), kTwo = real_t(2);
  const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cim = vindex(im, j, k, ny_total, nz_total);
  const long long cip = vindex(ip, j, k, ny_total, nz_total);
  const long long cjm = vindex(i, jm, k, ny_total, nz_total);
  const long long cjp = vindex(i, jp, k, ny_total, nz_total);
  const long long ckm = vindex(i, j, km, ny_total, nz_total);
  const long long ckp = vindex(i, j, kp, ny_total, nz_total);

  const real_t v5 = kHalf * (V[c] + V[cjm]);
  const real_t v6 = kHalf * (V[c] + V[cjp]);
  const real_t u1 = face_blend(dx2[jm], U[c], dx2[j], U[cjm], dmx2[j]);
  const real_t u2 = face_blend(dx2[jm], U[cip], dx2[j], U[vindex(ip, jm, k, ny_total, nz_total)], dmx2[j]);
  const real_t w3 = face_blend(dx2[jm], W[c], dx2[j], W[cjm], dmx2[j]);
  const real_t w4 = face_blend(dx2[jm], W[ckp], dx2[j], W[vindex(i, jm, kp, ny_total, nz_total)], dmx2[j]);

  const real_t dvdx1 = (V[c] - V[cim]) / dmx1[i];
  const real_t dvdx2 = (V[cip] - V[c]) / dmx1[ip];
  const real_t dvdz3 = (V[c] - V[ckm]) / dmx3[k];
  const real_t dvdz4 = (V[ckp] - V[c]) / dmx3[kp];
  const real_t dvdy5 = (V[c] - V[cjm]) / dx2[jm];
  const real_t dvdy6 = (V[cjp] - V[c]) / dx2[j];
  const real_t dudy1 = (U[c] - U[cjm]) / dmx2[j];
  const real_t dudy2 = (U[cip] - U[vindex(ip, jm, k, ny_total, nz_total)]) / dmx2[j];
  const real_t dwdy3 = (W[c] - W[cjm]) / dmx2[j];
  const real_t dwdy4 = (W[ckp] - W[vindex(i, jm, kp, ny_total, nz_total)]) / dmx2[j];

  const real_t xa = face_blend(dx2[jm], mu[cim], dx2[j], mu[vindex(im, jm, k, ny_total, nz_total)], dmx2[j]);
  const real_t xc = face_blend(dx2[jm], mu[c], dx2[j], mu[cjm], dmx2[j]);
  const real_t xb = face_blend(dx2[jm], mu[cip], dx2[j], mu[vindex(ip, jm, k, ny_total, nz_total)], dmx2[j]);
  const real_t mu1 = kHalf * (dx1[im] * xc + dx1[i] * xa) / dmx1[i];
  const real_t mu2 = kHalf * (dx1[ip] * xc + dx1[i] * xb) / dmx1[ip];
  const real_t za = face_blend(dx2[jm], mu[ckm], dx2[j], mu[vindex(i, jm, km, ny_total, nz_total)], dmx2[j]);
  const real_t zb = face_blend(dx2[jm], mu[ckp], dx2[j], mu[vindex(i, jm, kp, ny_total, nz_total)], dmx2[j]);
  const real_t muz3 = kHalf * (dx3[km] * xc + dx3[k] * za) / dmx3[k];
  const real_t muz4 = kHalf * (dx3[kp] * xc + dx3[k] * zb) / dmx3[kp];

  const real_t irc = face_blend(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);
  const real_t inh = kHalf * Cmu * irc;

  const real_t vX = (mu2 * dvdx2 - mu1 * dvdx1) / dx1[i];
  const real_t vZ = (muz4 * dvdz4 - muz3 * dvdz3) / dx3[k];
  const real_t vYo = (mu[c] * dvdy6 - mu[cjm] * dvdy5) / dmx2[j];
  const real_t vXc = (mu2 * dudy2 - mu1 * dudy1) / dx1[i];
  const real_t vZc = (muz4 * dwdy4 - muz3 * dwdy3) / dx3[k];
  real_t R = inh * (vX + vZ + kTwo * vYo + vXc + vZc);

  const real_t mACI = inh / dx1[i] * (mu2 / dmx1[ip] + mu1 / dmx1[i]) + kQ * (-u2 / dmx1[ip] + u1 / dmx1[i]);
  const real_t mAPI = -inh / dx1[i] * mu2 / dmx1[ip] + kQ * (u2 / dmx1[ip]);
  const real_t mAMI = -inh / dx1[i] * mu1 / dmx1[i] + kQ * (-u1 / dmx1[i]);
  const real_t mACK = inh / dx3[k] * (muz4 / dmx3[kp] + muz3 / dmx3[k]) + kQ * (-w4 / dmx3[kp] + w3 / dmx3[k]);
  const real_t mAPK = -inh / dx3[k] * muz4 / dmx3[kp] + kQ * (w4 / dmx3[kp]);
  const real_t mAMK = -inh / dx3[k] * muz3 / dmx3[k] + kQ * (-w3 / dmx3[k]);
  const real_t mACJ = inh / dmx2[j] * (mu[c] / dx2[j] + mu[cjm] / dx2[jm]) * kTwo
                      + kQ * (kHalf * dvdy6 + kHalf * dvdy5 - v6 / dx2[j] + v5 / dx2[jm]);
  const real_t mAPJ = -inh / dmx2[j] * mu[c] / dx2[j] * kTwo + kQ * (v6 / dx2[j] + kHalf * dvdy6);
  const real_t mAMJ = -inh / dmx2[j] * mu[cjm] / dx2[jm] * kTwo + kQ * (-v5 / dx2[jm] + kHalf * dvdy5);

  R -= (mAPI * V[cip] + mACI * V[c] + mAMI * V[cim] + mAPJ * V[cjp] + mACJ * V[c]
        + mAMJ * V[cjm] + mAPK * V[ckp] + mACK * V[c] + mAMK * V[ckm]);
  R -= (kQ * (u2 * dvdx2 + u1 * dvdx1) - inh / dx1[i] * (mu2 * dudy2 - mu1 * dudy1));
  R -= (kQ * (w3 * dvdz3 + w4 * dvdz4) - inh / dx3[k] * (muz4 * dwdy4 - muz3 * dwdy3));
  return R * dt;
}

template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t rhs_w_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* U, const real_t* V, const real_t* W, const MuArray mu,
    const RhoArray invrho, const real_t* dx1, const real_t* dmx1, const real_t* dx2,
    const real_t* dmx2, const real_t* dx3, const real_t* dmx3, const real_t Cmu,
    const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25), kTwo = real_t(2);
  const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cim = vindex(im, j, k, ny_total, nz_total);
  const long long cip = vindex(ip, j, k, ny_total, nz_total);
  const long long cjm = vindex(i, jm, k, ny_total, nz_total);
  const long long cjp = vindex(i, jp, k, ny_total, nz_total);
  const long long ckm = vindex(i, j, km, ny_total, nz_total);
  const long long ckp = vindex(i, j, kp, ny_total, nz_total);

  const real_t w5 = kHalf * (W[ckm] + W[c]);
  const real_t w6 = kHalf * (W[c] + W[ckp]);
  const real_t u1 = face_blend(dx3[km], U[c], dx3[k], U[ckm], dmx3[k]);
  const real_t u2 = face_blend(dx3[km], U[cip], dx3[k], U[vindex(ip, j, km, ny_total, nz_total)], dmx3[k]);
  const real_t v3 = face_blend(dx3[km], V[c], dx3[k], V[ckm], dmx3[k]);
  const real_t v4 = face_blend(dx3[km], V[cjp], dx3[k], V[vindex(i, jp, km, ny_total, nz_total)], dmx3[k]);

  const real_t dwdx1 = (W[c] - W[cim]) / dmx1[i];
  const real_t dwdx2 = (W[cip] - W[c]) / dmx1[ip];
  const real_t dwdz5 = (W[c] - W[ckm]) / dx3[km];
  const real_t dwdz6 = (W[ckp] - W[c]) / dx3[k];
  const real_t dwdy3 = (W[c] - W[cjm]) / dmx2[j];
  const real_t dwdy4 = (W[cjp] - W[c]) / dmx2[jp];
  const real_t dudz1 = (U[c] - U[ckm]) / dmx3[k];
  const real_t dudz2 = (U[cip] - U[vindex(ip, j, km, ny_total, nz_total)]) / dmx3[k];
  const real_t dvdz3 = (V[c] - V[ckm]) / dmx3[k];
  const real_t dvdz4 = (V[cjp] - V[vindex(i, jp, km, ny_total, nz_total)]) / dmx3[k];

  const real_t xa = face_blend(dx3[km], mu[cim], dx3[k], mu[vindex(im, j, km, ny_total, nz_total)], dmx3[k]);
  const real_t xc = face_blend(dx3[km], mu[c], dx3[k], mu[ckm], dmx3[k]);
  const real_t xb = face_blend(dx3[km], mu[cip], dx3[k], mu[vindex(ip, j, km, ny_total, nz_total)], dmx3[k]);
  const real_t mu1 = kHalf * (dx1[im] * xc + dx1[i] * xa) / dmx1[i];
  const real_t mu2 = kHalf * (dx1[ip] * xc + dx1[i] * xb) / dmx1[ip];
  const real_t ya = face_blend(dx3[km], mu[cjm], dx3[k], mu[vindex(i, jm, km, ny_total, nz_total)], dmx3[k]);
  const real_t yb = face_blend(dx3[km], mu[cjp], dx3[k], mu[vindex(i, jp, km, ny_total, nz_total)], dmx3[k]);
  const real_t muy5 = kHalf * (dx2[jm] * xc + dx2[j] * ya) / dmx2[j];
  const real_t muy6 = kHalf * (dx2[jp] * xc + dx2[j] * yb) / dmx2[jp];

  const real_t irc = face_blend(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]);
  const real_t inh = kHalf * Cmu * irc;

  const real_t vX = (mu2 * dwdx2 - mu1 * dwdx1) / dx1[i];
  const real_t vZo = (mu[c] * dwdz6 - mu[ckm] * dwdz5) / dmx3[k];
  const real_t vY = (muy6 * dwdy4 - muy5 * dwdy3) / dx2[j];
  const real_t vXc = (mu2 * dudz2 - mu1 * dudz1) / dx1[i];
  const real_t vYc = (muy6 * dvdz4 - muy5 * dvdz3) / dx2[j];
  real_t R = inh * (vX + kTwo * vZo + vY + vXc + vYc);  // 부력은 main 조립

  const real_t mACI = inh / dx1[i] * (mu2 / dmx1[ip] + mu1 / dmx1[i]) + kQ * (-u2 / dmx1[ip] + u1 / dmx1[i]);
  const real_t mAPI = -inh / dx1[i] * mu2 / dmx1[ip] + kQ * (u2 / dmx1[ip]);
  const real_t mAMI = -inh / dx1[i] * mu1 / dmx1[i] + kQ * (-u1 / dmx1[i]);
  const real_t mACK = inh / dmx3[k] * (mu[c] / dx3[k] + mu[ckm] / dx3[km]) * kTwo
                      + kQ * (kHalf * dwdz6 + kHalf * dwdz5 - w6 / dx3[k] + w5 / dx3[km]);
  const real_t mAPK = -inh / dmx3[k] * mu[c] / dx3[k] * kTwo + kQ * (w6 / dx3[k] + kHalf * dwdz6);
  const real_t mAMK = -inh / dmx3[k] * mu[ckm] / dx3[km] * kTwo + kQ * (-w5 / dx3[km] + kHalf * dwdz5);
  const real_t mACJ = inh / dx2[j] * (muy6 / dmx2[jp] + muy5 / dmx2[j]) + kQ * (-v4 / dmx2[jp] + v3 / dmx2[j]);
  const real_t mAPJ = -inh / dx2[j] * muy6 / dmx2[jp] + kQ * (v4 / dmx2[jp]);
  const real_t mAMJ = -inh / dx2[j] * muy5 / dmx2[j] + kQ * (-v3 / dmx2[j]);

  real_t sumM = mAPI * W[cip] + mACI * W[c] + mAMI * W[cim] + mAPJ * W[cjp]
                + mACJ * W[c] + mAMJ * W[cjm] + mAPK * W[ckp] + mACK * W[c]
                + mAMK * W[ckm];
  R -= sumM;
  R -= (kQ * (u2 * dwdx2 + u1 * dwdx1) - inh / dx1[i] * (mu2 * dudz2 - mu1 * dudz1));
  R -= (kQ * (v4 * dwdy4 + v3 * dwdy3) - inh / dx2[j] * (muy6 * dvdz4 - muy5 * dvdz3));
  return R * dt;
}

// ── ② 스윕 밴드 한 행 (참조 build_adi_bands_var 의 emit 1:1) ─────────────────
// own 축 체적가중 면 보간 (셀 필드 f 를 own-마이너스 면으로):
template <class Array>
MPMSTD_HD inline real_t own_face_blend(const Array f, const long long cell,
                                       const int own_index, const long long own_stride,
                                       const real_t* dx_own, const real_t* dmx_own) {
  return real_t(0.5)
         * (dx_own[own_index - 1] * f[cell] + dx_own[own_index] * f[cell - own_stride])
         / dmx_own[own_index];
}

template <class MuArray, class RhoArray>
MPMSTD_HD inline void sweep_band(
    const bool own, const bool z_staggered, const long long gi, const long long gm,
    const long long gp, const int a, const int own_index, const long long own_stride,
    const real_t* q, const real_t* q_advect, const MuArray mu, const RhoArray invrho,
    const real_t* dx_along, const real_t* dmx_along, const real_t* dx_own,
    const real_t* dmx_own, const real_t Cmu, const real_t dt, real_t& A_value,
    real_t& B_value, real_t& C_value) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25), kTwo = real_t(2);
  const real_t inh = kHalf * Cmu * dt;
  const real_t cq = kQ * dt;
  const real_t ir = own_face_blend(invrho, gi, own_index, own_stride, dx_own, dmx_own);

  real_t ad, cd;  // 확산
  if (own) {
    const real_t cmu = mu[gi];
    const real_t mmu = mu[gm];  // own-마이너스 면 μ = own 축 마이너스 이웃의 μ
    ad = -inh * ir / dmx_along[a] * mmu / dx_along[a - 1] * kTwo;
    cd = -inh * ir / dmx_along[a] * cmu / dx_along[a] * kTwo;
  } else {
    // 이중 체적가중 cross-면 μ: own 축으로 속도 면에 먼저, 스윕 축으로 다시
    const real_t muc = own_face_blend(mu, gi, own_index, own_stride, dx_own, dmx_own);
    const real_t mua = own_face_blend(mu, gm, own_index, own_stride, dx_own, dmx_own);
    const real_t mub = own_face_blend(mu, gp, own_index, own_stride, dx_own, dmx_own);
    const real_t mm = kHalf * (dx_along[a - 1] * muc + dx_along[a] * mua) / dmx_along[a];
    const real_t mp = kHalf * (dx_along[a + 1] * muc + dx_along[a] * mub) / dmx_along[a + 1];
    if (z_staggered) {
      ad = -inh * ir / dmx_along[a] * mm / dx_along[a - 1];
      cd = -inh * ir / dmx_along[a] * mp / dx_along[a];
    } else {
      ad = -inh * ir / dx_along[a] * mm / dmx_along[a];
      cd = -inh * ir / dx_along[a] * mp / dmx_along[a + 1];
    }
  }

  real_t ac, cc, bc;  // ¼ 준음해 대류
  if (own) {
    const real_t q5 = kHalf * (q_advect[gm] + q_advect[gi]);
    const real_t q6 = kHalf * (q_advect[gi] + q_advect[gp]);
    const real_t dq5 = (q[gi] - q[gm]) / dx_along[a - 1];
    const real_t dq6 = (q[gp] - q[gi]) / dx_along[a];
    ac = cq * (-q5 / dx_along[a - 1] + kHalf * dq5);
    cc = cq * (q6 / dx_along[a] + kHalf * dq6);
    bc = cq * (kHalf * dq6 + kHalf * dq5 - q6 / dx_along[a] + q5 / dx_along[a - 1]);
  } else {
    // cross: 이송 속도는 OWN 축 체적가중 (참조 w5/w6/v3/v4 와 같은 면 값)
    const real_t q5 = own_face_blend(q_advect, gi, own_index, own_stride, dx_own, dmx_own);
    const real_t q6 = own_face_blend(q_advect, gp, own_index, own_stride, dx_own, dmx_own);
    if (z_staggered) {
      ac = cq * (-q5 / dx_along[a - 1]);
      cc = cq * (q6 / dx_along[a]);
      bc = cq * (-q6 / dx_along[a] + q5 / dx_along[a - 1]);
    } else {
      ac = cq * (-q5 / dmx_along[a]);
      cc = cq * (q6 / dmx_along[a + 1]);
      bc = cq * (-q6 / dmx_along[a + 1] + q5 / dmx_along[a]);
    }
  }
  A_value = ad + ac;
  C_value = cd + cc;
  B_value = (real_t(1) - ad - cd) + bc;
}

// ── ③ 증분 커플링 (참조 couple_dd_* / block_couple_*_var 1:1) ────────────────
// 반환값만큼 대상에서 **빼면** 된다 (dV[c] -= ..., rhs_v[c] -= ...).

// M21: δU 가 dV 방정식 우변으로 (dV 스윕 전에 주입)
template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t couple_rhs_v_from_delta_u_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* delta_u, const real_t* V, const MuArray mu, const RhoArray invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t Cmu, const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25);
  const int im = i - 1, ip = i + 1, jm = j - 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cim = vindex(im, j, k, ny_total, nz_total);
  const long long cip = vindex(ip, j, k, ny_total, nz_total);
  const long long cjm = vindex(i, jm, k, ny_total, nz_total);

  const real_t xa = face_blend(dx2[jm], mu[cim], dx2[j], mu[vindex(im, jm, k, ny_total, nz_total)], dmx2[j]);
  const real_t xc = face_blend(dx2[jm], mu[c], dx2[j], mu[cjm], dmx2[j]);
  const real_t xb = face_blend(dx2[jm], mu[cip], dx2[j], mu[vindex(ip, jm, k, ny_total, nz_total)], dmx2[j]);
  const real_t mu1 = kHalf * (dx1[im] * xc + dx1[i] * xa) / dmx1[i];
  const real_t mu2 = kHalf * (dx1[ip] * xc + dx1[i] * xb) / dmx1[ip];
  const real_t irc = face_blend(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);
  const real_t inh = kHalf * Cmu * irc;

  const real_t dvdx1 = (V[c] - V[cim]) / dmx1[i];
  const real_t dvdx2 = (V[cip] - V[c]) / dmx1[ip];
  const real_t ddu1 = face_blend(dx2[jm], delta_u[c], dx2[j], delta_u[cjm], dmx2[j]);
  const real_t ddu2 = face_blend(dx2[jm], delta_u[cip], dx2[j],
                                 delta_u[vindex(ip, jm, k, ny_total, nz_total)], dmx2[j]);
  const real_t ddudy1 = (delta_u[c] - delta_u[cjm]) / dmx2[j];
  const real_t ddudy2 =
      (delta_u[cip] - delta_u[vindex(ip, jm, k, ny_total, nz_total)]) / dmx2[j];

  return dt * (kQ * (ddu2 * dvdx2 + ddu1 * dvdx1)
               - inh / dx1[i] * (mu2 * ddudy2 - mu1 * ddudy1));
}

// M31 + M32: δU·δV 가 dW 방정식 우변으로 (dW 스윕 전에 주입)
template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t couple_rhs_w_from_delta_uv_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* delta_u, const real_t* delta_v, const real_t* W, const MuArray mu,
    const RhoArray invrho, const real_t* dx1, const real_t* dmx1, const real_t* dx2,
    const real_t* dmx2, const real_t* dx3, const real_t* dmx3, const real_t Cmu,
    const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25);
  const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1, km = k - 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cim = vindex(im, j, k, ny_total, nz_total);
  const long long cip = vindex(ip, j, k, ny_total, nz_total);
  const long long cjm = vindex(i, jm, k, ny_total, nz_total);
  const long long cjp = vindex(i, jp, k, ny_total, nz_total);
  const long long ckm = vindex(i, j, km, ny_total, nz_total);

  const real_t irc = face_blend(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]);
  const real_t inh = kHalf * Cmu * irc;
  const real_t xc = face_blend(dx3[km], mu[c], dx3[k], mu[ckm], dmx3[k]);

  // M31: δU
  const real_t ddu1 = face_blend(dx3[km], delta_u[c], dx3[k], delta_u[ckm], dmx3[k]);
  const real_t ddu2 = face_blend(dx3[km], delta_u[cip], dx3[k],
                                 delta_u[vindex(ip, j, km, ny_total, nz_total)], dmx3[k]);
  const real_t dWdx1 = (W[c] - W[cim]) / dmx1[i];
  const real_t dWdx2 = (W[cip] - W[c]) / dmx1[ip];
  const real_t ddudz1 = (delta_u[c] - delta_u[ckm]) / dmx3[k];
  const real_t ddudz2 =
      (delta_u[cip] - delta_u[vindex(ip, j, km, ny_total, nz_total)]) / dmx3[k];
  const real_t xa = face_blend(dx3[km], mu[cim], dx3[k], mu[vindex(im, j, km, ny_total, nz_total)], dmx3[k]);
  const real_t xb = face_blend(dx3[km], mu[cip], dx3[k], mu[vindex(ip, j, km, ny_total, nz_total)], dmx3[k]);
  const real_t mu1 = kHalf * (dx1[im] * xc + dx1[i] * xa) / dmx1[i];
  const real_t mu2 = kHalf * (dx1[ip] * xc + dx1[i] * xb) / dmx1[ip];
  real_t amount = dt * (kQ * (ddu2 * dWdx2 + ddu1 * dWdx1)
                        - inh / dx1[i] * (mu2 * ddudz2 - mu1 * ddudz1));

  // M32: δV
  const real_t dvm5 = face_blend(dx3[km], delta_v[c], dx3[k], delta_v[ckm], dmx3[k]);
  const real_t dvm6 = face_blend(dx3[km], delta_v[cjp], dx3[k],
                                 delta_v[vindex(i, jp, km, ny_total, nz_total)], dmx3[k]);
  const real_t dWdy5 = (W[c] - W[cjm]) / dmx2[j];
  const real_t dWdy6 = (W[cjp] - W[c]) / dmx2[jp];
  const real_t ddVz5 = (delta_v[c] - delta_v[ckm]) / dmx3[k];
  const real_t ddVz6 =
      (delta_v[cjp] - delta_v[vindex(i, jp, km, ny_total, nz_total)]) / dmx3[k];
  const real_t ya = face_blend(dx3[km], mu[cjm], dx3[k], mu[vindex(i, jm, km, ny_total, nz_total)], dmx3[k]);
  const real_t yb = face_blend(dx3[km], mu[cjp], dx3[k], mu[vindex(i, jp, km, ny_total, nz_total)], dmx3[k]);
  const real_t muy5 = kHalf * (dx2[jm] * xc + dx2[j] * ya) / dmx2[j];
  const real_t muy6 = kHalf * (dx2[jp] * xc + dx2[j] * yb) / dmx2[jp];
  amount += dt * (kQ * (dvm5 * dWdy5 + dvm6 * dWdy6)
                  - inh / dx2[j] * (muy6 * ddVz6 - muy5 * ddVz5));
  return amount;
}

// M23 (사후): δW 가 δV 를 보정
template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t couple_delta_v_from_delta_w_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* delta_w, const real_t* V, const MuArray mu, const RhoArray invrho,
    const real_t* dx2, const real_t* dmx2, const real_t* dx3, const real_t* dmx3,
    const real_t Cmu, const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25);
  const int jm = j - 1, km = k - 1, kp = k + 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cjm = vindex(i, jm, k, ny_total, nz_total);
  const long long ckm = vindex(i, j, km, ny_total, nz_total);
  const long long ckp = vindex(i, j, kp, ny_total, nz_total);

  const real_t ddw3 = face_blend(dx2[jm], delta_w[c], dx2[j], delta_w[cjm], dmx2[j]);
  const real_t ddw4 = face_blend(dx2[jm], delta_w[ckp], dx2[j],
                                 delta_w[vindex(i, jm, kp, ny_total, nz_total)], dmx2[j]);
  const real_t ddwdy3 = (delta_w[c] - delta_w[cjm]) / dmx2[j];
  const real_t ddwdy4 =
      (delta_w[ckp] - delta_w[vindex(i, jm, kp, ny_total, nz_total)]) / dmx2[j];
  const real_t dvdz3 = (V[c] - V[ckm]) / dmx3[k];
  const real_t dvdz4 = (V[ckp] - V[c]) / dmx3[kp];

  const real_t xc = face_blend(dx2[jm], mu[c], dx2[j], mu[cjm], dmx2[j]);
  const real_t za = face_blend(dx2[jm], mu[ckm], dx2[j], mu[vindex(i, jm, km, ny_total, nz_total)], dmx2[j]);
  const real_t zb = face_blend(dx2[jm], mu[ckp], dx2[j], mu[vindex(i, jm, kp, ny_total, nz_total)], dmx2[j]);
  const real_t muz3 = kHalf * (dx3[km] * xc + dx3[k] * za) / dmx3[k];
  const real_t muz4 = kHalf * (dx3[kp] * xc + dx3[k] * zb) / dmx3[kp];
  const real_t irc = face_blend(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);

  return dt * (kQ * (ddw3 * dvdz3 + ddw4 * dvdz4)
               - kHalf * Cmu * irc / dx3[k] * (muz4 * ddwdy4 - muz3 * ddwdy3));
}

// M12 + M13 (사후): δW·δV 가 δU 를 보정
template <class MuArray, class RhoArray>
MPMSTD_HD inline real_t couple_delta_u_from_delta_vw_cell(
    const int i, const int j, const int k, const int ny_total, const int nz_total,
    const real_t* delta_v, const real_t* delta_w, const real_t* U, const MuArray mu,
    const RhoArray invrho, const real_t* dx1, const real_t* dmx1, const real_t* dx2,
    const real_t* dmx2, const real_t* dx3, const real_t* dmx3, const real_t Cmu,
    const real_t dt) {
  const real_t kHalf = real_t(0.5), kQ = real_t(0.25);
  const int im = i - 1, jm = j - 1, jp = j + 1, km = k - 1, kp = k + 1;
  const long long c = vindex(i, j, k, ny_total, nz_total);
  const long long cim = vindex(im, j, k, ny_total, nz_total);
  const long long cjp = vindex(i, jp, k, ny_total, nz_total);
  const long long ckp = vindex(i, j, kp, ny_total, nz_total);
  const real_t irc = face_blend(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]);

  // M12: δW 가 ∂U/∂z 를 통해
  const real_t dWm5 = face_blend(dx1[i], delta_w[cim], dx1[im], delta_w[c], dmx1[i]);
  const real_t dWm6 = face_blend(dx1[i], delta_w[vindex(im, j, kp, ny_total, nz_total)],
                                 dx1[im], delta_w[ckp], dmx1[i]);
  const real_t dUdz5 = (U[c] - U[vindex(i, j, km, ny_total, nz_total)]) / dmx3[k];
  const real_t dUdz6 = (U[ckp] - U[c]) / dmx3[kp];
  const real_t dWmx5 = (delta_w[c] - delta_w[cim]) / dmx1[i];
  const real_t dWmx6 =
      (delta_w[ckp] - delta_w[vindex(im, j, kp, ny_total, nz_total)]) / dmx1[i];
  const real_t za = face_blend(dx1[im], mu[vindex(i, j, km, ny_total, nz_total)], dx1[i],
                               mu[vindex(im, j, km, ny_total, nz_total)], dmx1[i]);
  const real_t zc = face_blend(dx1[im], mu[c], dx1[i], mu[cim], dmx1[i]);
  const real_t zb = face_blend(dx1[im], mu[ckp], dx1[i],
                               mu[vindex(im, j, kp, ny_total, nz_total)], dmx1[i]);
  const real_t muz5 = kHalf * (dx3[km] * zc + dx3[k] * za) / dmx3[k];
  const real_t muz6 = kHalf * (dx3[kp] * zc + dx3[k] * zb) / dmx3[kp];
  real_t amount = dt * (kQ * (dWm6 * dUdz6 + dWm5 * dUdz5)
                        - kHalf * Cmu * irc / dx3[k] * (muz6 * dWmx6 - muz5 * dWmx5));

  // M13: δV 가 ∂U/∂y 를 통해
  const real_t dVm3 = face_blend(dx1[i], delta_v[cim], dx1[im], delta_v[c], dmx1[i]);
  const real_t dVm4 = face_blend(dx1[i], delta_v[vindex(im, jp, k, ny_total, nz_total)],
                                 dx1[im], delta_v[cjp], dmx1[i]);
  const real_t dUdy3 = (U[c] - U[vindex(i, jm, k, ny_total, nz_total)]) / dmx2[j];
  const real_t dUdy4 = (U[cjp] - U[c]) / dmx2[jp];
  const real_t dVmx3 = (delta_v[c] - delta_v[cim]) / dmx1[i];
  const real_t dVmx4 =
      (delta_v[cjp] - delta_v[vindex(im, jp, k, ny_total, nz_total)]) / dmx1[i];
  const real_t ya = face_blend(dx1[im], mu[vindex(i, jm, k, ny_total, nz_total)], dx1[i],
                               mu[vindex(im, jm, k, ny_total, nz_total)], dmx1[i]);
  const real_t yb = face_blend(dx1[im], mu[cjp], dx1[i],
                               mu[vindex(im, jp, k, ny_total, nz_total)], dmx1[i]);
  const real_t muy3 = kHalf * (dx2[jm] * zc + dx2[j] * ya) / dmx2[j];
  const real_t muy4 = kHalf * (dx2[jp] * zc + dx2[j] * yb) / dmx2[jp];
  amount += dt * (kQ * (dVm4 * dUdy4 + dVm3 * dUdy3)
                  - kHalf * Cmu * irc / dx2[j] * (muy4 * dVmx4 - muy3 * dVmx3));
  return amount;
}

}  // namespace mpmstd::equation::momentum::stencil
