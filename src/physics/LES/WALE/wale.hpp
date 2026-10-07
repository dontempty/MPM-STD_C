#pragma once
// WALE 와점성 (Nicoud & Ducros 1999) — 식은 wale_stencil.hpp 한 곳 (CPU wale.cpp / GPU wale.cu).
// mu_hat ← 1 + ν_t/ν (내부 셀). 뒤에 les::synchronize_viscosity_ratio 로 ghost 를 맞춘다.
//   c_wale = C_w (논문 권장 0.5; C_w² ≈ 10.6 C_s²)
#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics::les {

void calculate_eddy_viscosity_wale(field::Field& mu_hat, const field::Field& U,
                                   const field::Field& V, const field::Field& W,
                                   const grid::Grid& grid, const parallel::Subdomain& sub,
                                   real_t nu, real_t c_wale);

}  // namespace mpmstd::physics::les
