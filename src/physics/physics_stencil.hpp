#pragma once
// 물리 항 스텐실 — CPU/GPU 쌍둥이 공용 (D3).

#include "common/hd.hpp"
#include "common/types.hpp"
#include "field/view.hpp"

namespace mpmstd::physics::stencil {

// T 를 W 의 z-면 위치로 체적가중 보간 (buoyancy.hpp 의 ⚠ 참조).
MPMSTD_HD inline real_t temperature_at_z_face(const field::ConstView T, const int i,
                                              const int j, const int k,
                                              const real_t* dz, const real_t* dmz) {
  return real_t(0.5) * (dz[k - 1] * T(i, j, k) + dz[k] * T(i, j, k - 1)) / dmz[k];
}

// x-면(U 위치) 온도 — DHVC 는 중력이 x(연직)라 부력이 U 에 실린다.
MPMSTD_HD inline real_t temperature_at_x_face(const field::ConstView T, const int i,
                                              const int j, const int k,
                                              const real_t* dx, const real_t* dmx) {
  return real_t(0.5) * (dx[i - 1] * T(i, j, k) + dx[i] * T(i - 1, j, k)) / dmx[i];
}

}  // namespace mpmstd::physics::stencil
