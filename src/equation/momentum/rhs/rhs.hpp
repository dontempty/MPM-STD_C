#pragma once
// 충실(참조 1:1) 운동량 stage-1 RHS — 물성 축으로 두 진입점을 제공한다:
//   *_variable_property  : 물성 배열(μ̂, 1/ρ̂) 사용 — NOB (rbc_nob)
//   *_constant_property  : 물성 배열 없음(UnitArray 접힘), 스칼라 ν — OB (channel·DHVC)
// 수식 정본은 momentum_stencil.hpp 하나 (물성 접근자 템플릿) — 두 경로는
// 같은 식의 인스턴스화라 "가변+1배열 = 상수" 비트 일치가 보장된다.
// 가변 점성(NOB) 운동량의 stage-1 명시 우변 — rhs += dt·(R − M·qⁿ − 북키핑).
//
// OB(rhs.hpp)와 역할은 같지만 식이 다르다: own 방향 확산 ×2, cross-stress
// (전치 응력) 포함, 이중 체적가중 면 μ, 1/ρ̂ 면 보간. 압력·부력은 여기 없다 —
// main 이 뒤에 += 로 조립한다 (pressure::add_gradient_variable_property, buoyancy).
// 상수물성 극한(μ̂≡1, 1/ρ̂≡1)에서 cross-stress 는 비압축 연속으로 소거되는
// 항이라 OB 와 같은 물리로 환원된다 (이산적으로는 참조 스킴 그대로).
//
// Cmu = √(Pr/Ra) (자유낙하 무차원의 점성 계수; 참조 Cmu).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

void add_advection_diffusion_u_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

void add_advection_diffusion_v_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

void add_advection_diffusion_w_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

// ── 전용 RHS 빌드 (2026-10-17, 운동량 RHS 최적화 3): rbc_nob 의 stage-1 우변 전체를 성분당 커널 한 번으로 —
//    rhs_q = dt·R_q(이송·확산)  [+ 부력 (W 만)]  − dt·(1/ρ̂)∇P.  main 이 fill → add ×3 → 부력 → 구배의 8 패스로
//    하던 것을 3 패스로. 덧셈 순서는 그 순서 그대로. GPU 판은 j 방향 스레드 coarsening(최적화 2)과 격자 역수(최적화 1)도 쓴다.
//    CPU 판(rhs.cpp)은 기존 호출을 그 순서로 부르는 참조 구현.
void build_variable_property_rhs(field::Field& rhs_u, field::Field& rhs_v, field::Field& rhs_w,
                                 const field::Field& U, const field::Field& V,
                                 const field::Field& W, const field::Field& T,
                                 const field::Field& P, const field::Field& viscosity,
                                 const field::Field& inverse_density, const grid::Grid& grid,
                                 const parallel::Subdomain& sub, real_t Cmu,
                                 real_t density_quadratic_ratio, real_t delta_temperature,
                                 real_t dt);

// ── 상수물성 판 (plan §18-1) — 물성 배열 없이 같은 수식 (viscosity = 스칼라 ν).
void add_advection_diffusion_u_constant_property(field::Field& rhs,
                                                 const field::Field& U,
                                                 const field::Field& V,
                                                 const field::Field& W,
                                                 const grid::Grid& grid,
                                                 const parallel::Subdomain& sub,
                                                 real_t viscosity, real_t dt);
void add_advection_diffusion_v_constant_property(field::Field& rhs,
                                                 const field::Field& U,
                                                 const field::Field& V,
                                                 const field::Field& W,
                                                 const grid::Grid& grid,
                                                 const parallel::Subdomain& sub,
                                                 real_t viscosity, real_t dt);
void add_advection_diffusion_w_constant_property(field::Field& rhs,
                                                 const field::Field& U,
                                                 const field::Field& V,
                                                 const field::Field& W,
                                                 const grid::Grid& grid,
                                                 const parallel::Subdomain& sub,
                                                 real_t viscosity, real_t dt);

}  // namespace mpmstd::equation::momentum
