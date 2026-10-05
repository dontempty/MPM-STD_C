#pragma once
// 이류–확산 수송 스텐실의 정본 — **운동량과 에너지(스칼라)가 같은 식을 쓴다**.
// CPU 루프(.cpp)와 GPU 커널(.cu) 전부가 이 함수들을 부른다 (D3: 수식은 한 곳).
//   * diffusion            : ∇²q (계수 없이 — 호출부가 ν·dt 또는 α·dt 를 곱한다)
//   * convection_divergence: ∇·(u q) (보존형 면 플럭스 — MAC 면 속도 × 2점 평균,
//                            셀의 +면 = 이웃 인덱스의 면 속도, conventions §3)
//   * advection_diffusion  : 위 둘의 운동량/에너지 공용 합성
// w_is_z_face 는 z 라플라시안의 메트릭(셀 중심 U·V·T vs z-면 W)을 고른다.

#include "common/hd.hpp"
#include "common/types.hpp"
#include "field/view.hpp"

namespace mpmstd::equation::stencil {

MPMSTD_HD inline real_t diffusion(const field::ConstView q, const int i, const int j,
                                  const int k, const real_t* dx_x, const real_t* dmx_x,
                                  const real_t* dx_y, const real_t* dmx_y,
                                  const real_t* dx_z, const real_t* dmx_z,
                                  const bool w_is_z_face) {
  const real_t laplacian_x = ((q(i + 1, j, k) - q(i, j, k)) / dmx_x[i + 1]
                              - (q(i, j, k) - q(i - 1, j, k)) / dmx_x[i]) / dx_x[i];
  const real_t laplacian_y = ((q(i, j + 1, k) - q(i, j, k)) / dmx_y[j + 1]
                              - (q(i, j, k) - q(i, j - 1, k)) / dmx_y[j]) / dx_y[j];
  const real_t laplacian_z =
      w_is_z_face
          ? ((q(i, j, k + 1) - q(i, j, k)) / dx_z[k]
             - (q(i, j, k) - q(i, j, k - 1)) / dx_z[k - 1]) / dmx_z[k]
          : ((q(i, j, k + 1) - q(i, j, k)) / dmx_z[k + 1]
             - (q(i, j, k) - q(i, j, k - 1)) / dmx_z[k]) / dx_z[k];
  return laplacian_x + laplacian_y + laplacian_z;
}

MPMSTD_HD inline real_t convection_divergence(
    const field::ConstView q, const field::ConstView U, const field::ConstView V,
    const field::ConstView W, const int i, const int j, const int k,
    const real_t* dx_x, const real_t* dx_y, const real_t* dx_z) {
  const real_t half = real_t(0.5);
  const real_t flux_x_plus = U(i + 1, j, k) * half * (q(i, j, k) + q(i + 1, j, k));
  const real_t flux_x_minus = U(i, j, k) * half * (q(i - 1, j, k) + q(i, j, k));
  const real_t flux_y_plus = V(i, j + 1, k) * half * (q(i, j, k) + q(i, j + 1, k));
  const real_t flux_y_minus = V(i, j, k) * half * (q(i, j - 1, k) + q(i, j, k));
  const real_t flux_z_plus = W(i, j, k + 1) * half * (q(i, j, k) + q(i, j, k + 1));
  const real_t flux_z_minus = W(i, j, k) * half * (q(i, j, k - 1) + q(i, j, k));
  return (flux_x_plus - flux_x_minus) / dx_x[i]
         + (flux_y_plus - flux_y_minus) / dx_y[j]
         + (flux_z_plus - flux_z_minus) / dx_z[k];
}

// 명시 RHS 증분: 계수·∇²q − dt·∇·(u q)  (운동량: 계수 = ν·dt / 델타형 전체)
MPMSTD_HD inline real_t advection_diffusion(
    const field::ConstView q, const field::ConstView U, const field::ConstView V,
    const field::ConstView W, const int i, const int j, const int k,
    const real_t* dx_x, const real_t* dmx_x, const real_t* dx_y, const real_t* dmx_y,
    const real_t* dx_z, const real_t* dmx_z, const real_t nu_dt, const real_t dt,
    const bool w_is_z_face) {
  return nu_dt * diffusion(q, i, j, k, dx_x, dmx_x, dx_y, dmx_y, dx_z, dmx_z,
                           w_is_z_face)
         - dt * convection_divergence(q, U, V, W, i, j, k, dx_x, dx_y, dx_z);
}

}  // namespace mpmstd::equation::stencil
