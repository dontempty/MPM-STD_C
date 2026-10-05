#pragma once
// 운동량 전용 수식의 정본(밴드·블록 커플링) — CPU 루프와 GPU
// 커널(.cu 쌍둥이)이 **같은 함수**를 부른다 (설계 규칙 D3: 수식은 한 곳에만;
// MPM-STD_C 의 scalar_var_stencil.hpp 패턴). 인자는 전부 스칼라·POD 뷰·원시
// 포인터다 — 그래서 __host__ __device__ 가 된다 (common/hd.hpp).
// 물리 설명(CN ½ 확산·¼ 준음해 대류·z-면 메트릭·커플링)은 각 모듈 헤더 참조.

#include "common/hd.hpp"
#include "common/types.hpp"
#include "equation/transport_stencil.hpp"  // advection_diffusion — 수송 공용 정본
#include "field/view.hpp"

namespace mpmstd::equation::momentum::stencil {

// 음해 밴드 한 행 (I − ½dtν·L_d − ¼dt·C_d). 방향별 함수 — D13(축 인자 금지).
struct Band {
  real_t a, b, c;
};

MPMSTD_HD inline Band band_x(const field::ConstView adv, const int i, const int j,
                             const int k, const real_t* dx, const real_t* dmx,
                             const real_t coeff, const real_t cconv) {
  const real_t av = -coeff / (dx[i] * dmx[i]);
  const real_t cv = -coeff / (dx[i] * dmx[i + 1]);
  const real_t u_minus = real_t(0.5) * (adv(i - 1, j, k) + adv(i, j, k));
  const real_t u_plus = real_t(0.5) * (adv(i, j, k) + adv(i + 1, j, k));
  const real_t ac = -cconv * u_minus / dmx[i];
  const real_t cc = cconv * u_plus / dmx[i + 1];
  const real_t bc = -cconv * (-u_plus / dmx[i + 1] + u_minus / dmx[i]);
  return {av + ac, (1 - av - cv) + bc, cv + cc};
}

MPMSTD_HD inline Band band_y(const field::ConstView adv, const int i, const int j,
                             const int k, const real_t* dy, const real_t* dmy,
                             const real_t coeff, const real_t cconv) {
  const real_t av = -coeff / (dy[j] * dmy[j]);
  const real_t cv = -coeff / (dy[j] * dmy[j + 1]);
  const real_t v_minus = real_t(0.5) * (adv(i, j - 1, k) + adv(i, j, k));
  const real_t v_plus = real_t(0.5) * (adv(i, j, k) + adv(i, j + 1, k));
  const real_t ac = -cconv * v_minus / dmy[j];
  const real_t cc = cconv * v_plus / dmy[j + 1];
  const real_t bc = -cconv * (-v_plus / dmy[j + 1] + v_minus / dmy[j]);
  return {av + ac, (1 - av - cv) + bc, cv + cc};
}

MPMSTD_HD inline Band band_z_cell(const field::ConstView adv, const int i, const int j,
                                  const int k, const real_t* dz, const real_t* dmz,
                                  const real_t coeff, const real_t cconv) {
  const real_t av = -coeff / (dz[k] * dmz[k]);
  const real_t cv = -coeff / (dz[k] * dmz[k + 1]);
  const real_t w_minus = real_t(0.5) * (adv(i, j, k - 1) + adv(i, j, k));
  const real_t w_plus = real_t(0.5) * (adv(i, j, k) + adv(i, j, k + 1));
  const real_t ac = -cconv * w_minus / dmz[k];
  const real_t cc = cconv * w_plus / dmz[k + 1];
  const real_t bc = -cconv * (-w_plus / dmz[k + 1] + w_minus / dmz[k]);
  return {av + ac, (1 - av - cv) + bc, cv + cc};
}

// z-면 변수(W): 제어체적 = center_distance, 이웃 거리 = cell_width.
MPMSTD_HD inline Band band_z_face(const field::ConstView adv, const int i, const int j,
                                  const int k, const real_t* dz, const real_t* dmz,
                                  const real_t coeff, const real_t cconv) {
  const real_t av = -coeff / (dmz[k] * dz[k - 1]);
  const real_t cv = -coeff / (dmz[k] * dz[k]);
  const real_t w_minus = real_t(0.5) * (adv(i, j, k - 1) + adv(i, j, k));
  const real_t w_plus = real_t(0.5) * (adv(i, j, k) + adv(i, j, k + 1));
  const real_t ac = -cconv * w_minus / dz[k - 1];
  const real_t cc = cconv * w_plus / dz[k];
  const real_t bc = -cconv * (-w_plus / dz[k] + w_minus / dz[k - 1]);
  return {av + ac, (1 - av - cv) + bc, cv + cc};
}

// 블록 커플링 (상수 점성 — 대류만; MPM-STD_C block_couple_* 1:1).
// δV 가 받는 M23·δW 항 — 반환값을 dt 배 해서 δV 에서 뺀다.
MPMSTD_HD inline real_t couple_dv_from_dw(const field::ConstView dW,
                                          const field::ConstView V, const int i,
                                          const int j, const int k, const real_t* dy,
                                          const real_t* dmy, const real_t* dmz) {
  const real_t dw_minus = real_t(0.5)
      * (dy[j - 1] * dW(i, j, k) + dy[j] * dW(i, j - 1, k)) / dmy[j];
  const real_t dw_plus = real_t(0.5)
      * (dy[j - 1] * dW(i, j, k + 1) + dy[j] * dW(i, j - 1, k + 1)) / dmy[j];
  const real_t dvdz_minus = (V(i, j, k) - V(i, j, k - 1)) / dmz[k];
  const real_t dvdz_plus = (V(i, j, k + 1) - V(i, j, k)) / dmz[k + 1];
  return real_t(0.25) * (dw_minus * dvdz_minus + dw_plus * dvdz_plus);
}

// δU 가 받는 M12·δV + M13·δW 항 (보정된 δV 를 쓸 것 — solve.hpp 순서 주석).
MPMSTD_HD inline real_t couple_du_from_dv_dw(
    const field::ConstView dV, const field::ConstView dW, const field::ConstView U,
    const int i, const int j, const int k, const real_t* dx, const real_t* dmx,
    const real_t* dmy, const real_t* dmz) {
  const real_t dv_minus = real_t(0.5)
      * (dx[i - 1] * dV(i, j, k) + dx[i] * dV(i - 1, j, k)) / dmx[i];
  const real_t dv_plus = real_t(0.5)
      * (dx[i - 1] * dV(i, j + 1, k) + dx[i] * dV(i - 1, j + 1, k)) / dmx[i];
  const real_t dudy_minus = (U(i, j, k) - U(i, j - 1, k)) / dmy[j];
  const real_t dudy_plus = (U(i, j + 1, k) - U(i, j, k)) / dmy[j + 1];
  const real_t dw_minus = real_t(0.5)
      * (dx[i - 1] * dW(i, j, k) + dx[i] * dW(i - 1, j, k)) / dmx[i];
  const real_t dw_plus = real_t(0.5)
      * (dx[i - 1] * dW(i, j, k + 1) + dx[i] * dW(i - 1, j, k + 1)) / dmx[i];
  const real_t dudz_minus = (U(i, j, k) - U(i, j, k - 1)) / dmz[k];
  const real_t dudz_plus = (U(i, j, k + 1) - U(i, j, k)) / dmz[k + 1];
  return real_t(0.25) * ((dv_minus * dudy_minus + dv_plus * dudy_plus)
                         + (dw_minus * dudz_minus + dw_plus * dudz_plus));
}

}  // namespace mpmstd::equation::momentum::stencil
