#pragma once
// 압력(투영) 수식의 정본 — CPU/GPU 쌍둥이가 같은 함수를 쓴다 (D3).
// 설명은 pressure/system.hpp·rhs/rhs.hpp 참조.

#include "common/hd.hpp"
#include "common/types.hpp"
#include "field/view.hpp"

namespace mpmstd::equation::pressure::stencil {

// ∇·u |_c  (MAC: +면 = 이웃 인덱스의 면 속도)
MPMSTD_HD inline real_t divergence(const field::ConstView u, const field::ConstView v,
                                   const field::ConstView w, const int i, const int j,
                                   const int k, const real_t dx, const real_t dy,
                                   const real_t dz) {
  return (u(i + 1, j, k) - u(i, j, k)) / dx + (v(i, j + 1, k) - v(i, j, k)) / dy
         + (w(i, j, k + 1) - w(i, j, k)) / dz;
}

// 면 위치의 staggered 압력 구배 (분모 = 중심간거리)
MPMSTD_HD inline real_t gradient_x(const field::ConstView p, const int i, const int j,
                                   const int k, const real_t dmx) {
  return (p(i, j, k) - p(i - 1, j, k)) / dmx;
}
MPMSTD_HD inline real_t gradient_y(const field::ConstView p, const int i, const int j,
                                   const int k, const real_t dmy) {
  return (p(i, j, k) - p(i, j - 1, k)) / dmy;
}
MPMSTD_HD inline real_t gradient_z(const field::ConstView p, const int i, const int j,
                                   const int k, const real_t dmz) {
  return (p(i, j, k) - p(i, j, k - 1)) / dmz;
}

// (kx, ky) 모드의 z 삼중대각 한 행 — 벽 Neumann 은 접기(fold_*)로 들어온다.
// (0,0) 특이 모드의 항등행 핀은 호출부(두 쌍둥이의 같은 if)가 처리한다.
struct Band {
  real_t a, b, c;
};

MPMSTD_HD inline Band z_mode_band(const real_t dz, const real_t dmz_minus,
                                  const real_t dmz_plus, const real_t lambda,
                                  const bool fold_bottom, const bool fold_top) {
  const real_t a = fold_bottom ? real_t(0) : 1 / (dz * dmz_minus);
  const real_t c = fold_top ? real_t(0) : 1 / (dz * dmz_plus);
  return {a, -(a + c) - lambda, c};
}

}  // namespace mpmstd::equation::pressure::stencil
