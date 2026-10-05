#pragma once
// CFL 시간 간격 — MPM-STD_C driver/cfl 1:1.
//   speed = 전역 max( |U|/dx + |V|/dy + |W|/dz ),  dt = min(max_cfl/speed, dt_cap)
// ⚠ 대류가 명시(explicit)라 dt 는 반드시 CFL 을 따라야 한다 — dt 를 dt_cap 에
//   고정(바닥)하면 속도가 자라는 순간 발산한다 (MPM-STD_C Report 18 교훈;
//   dt_cap 은 상한일 뿐이다).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::time_integration {

real_t compute_cfl_dt(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub, real_t max_cfl, real_t dt_cap);

}  // namespace mpmstd::time_integration
