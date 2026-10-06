#pragma once
// Const-Vreman 와점성 — 참조 Fortran MPM-STD `Const-Vreman` 1:1 (vreman_stencil.hpp).
// mu_hat ← 1 + ν_t/ν (내부 셀). 뒤에 les::synchronize_viscosity_ratio 로 ghost 를 맞춘다.
//   c_vreman = 2.5·C_s² (Fortran: C_s = 0.12 → 0.036; Vreman 원논문 C_s 0.17 → 0.07)
#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics::les {

void calculate_eddy_viscosity_constant_vreman(field::Field& mu_hat, const field::Field& U,
                                              const field::Field& V, const field::Field& W,
                                              const grid::Grid& grid, const parallel::Subdomain& sub,
                                              real_t nu, real_t c_vreman);

}  // namespace mpmstd::physics::les
