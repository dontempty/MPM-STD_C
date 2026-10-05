#pragma once
// 가변 점성(NOB) 운동량의 stage-1 명시 우변 — rhs += dt·(R − M·qⁿ − 북키핑).
//
// OB(rhs.hpp)와 역할은 같지만 식이 다르다: own 방향 확산 ×2, cross-stress
// (전치 응력) 포함, 이중 체적가중 면 μ, 1/ρ̂ 면 보간. 압력·부력은 여기 없다 —
// main 이 뒤에 += 로 조립한다 (pressure::add_gradient_variable, buoyancy).
// 상수물성 극한(μ̂≡1, 1/ρ̂≡1)에서 cross-stress 는 비압축 연속으로 소거되는
// 항이라 OB 와 같은 물리로 환원된다 (이산적으로는 참조 스킴 그대로).
//
// Cmu = √(Pr/Ra) (자유낙하 무차원의 점성 계수; 참조 Cmu).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

void add_advection_diffusion_u_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

void add_advection_diffusion_v_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

void add_advection_diffusion_w_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

}  // namespace mpmstd::equation::momentum
