#pragma once
// 에너지(스칼라) 전용 수식 — CN ½ 확산 밴드 (대류는 명시라 밴드에 없다).
// CPU(system.cpp)와 GPU(system.cu)가 같이 쓴다 (D3). 수송 항은
// equation/transport_stencil.hpp 공용.

#include "common/hd.hpp"
#include "common/types.hpp"

namespace mpmstd::equation::scalar::stencil {

struct Band {
  real_t a, b, c;
};

// (I − ½dtα L_d) 의 한 행 — 스칼라 인자만 (D13 허용 형태).
MPMSTD_HD inline Band diffusion_band(const real_t half_dt_alpha, const real_t dx,
                                     const real_t dmx_minus, const real_t dmx_plus) {
  const real_t a = -half_dt_alpha / (dx * dmx_minus);
  const real_t c = -half_dt_alpha / (dx * dmx_plus);
  return {a, 1 - a - c, c};
}

}  // namespace mpmstd::equation::scalar::stencil
