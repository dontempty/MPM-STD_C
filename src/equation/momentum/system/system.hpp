#pragma once
// 가변 점성(NOB) 운동량의 스윕별 밴드 조립 — 참조 build_adi_bands_var 1:1.
//
// 스윕 방향은 함수 이름(x/y/z — D13), 성분은 own_axis(0=U, 1=V, 2=W)로 받는다
// (참조 1:1 이식 커널의 own 축 인자 — 밴드 수식은 own/cross 분기만 다르고
// momentum_stencil.hpp 의 sweep_band 한 곳에 있다).
//   * own 스윕: ×2 확산(셀 μ) + ¼ 대류 + ½∂q 자기미분 (dx 메트릭)
//   * cross 스윕: ×1 확산(이중 체적가중 면 μ) + ¼ 대류 (dmx 메트릭, 이송
//     속도는 own 축 체적가중 면 값)
// D 는 source(이전 단계의 증분 또는 stage-1 rhs)를 그대로 팩한다.
// 벽 닫기(fold/핀)는 호출자(solve_variable_property)가 OB 와 동일하게 적용한다.

#include "common/types.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "equation/band_layout.hpp"

namespace mpmstd::equation::momentum {

// source_layout: D 로 복사하는 source 가 전체 모양 필드인지, 직전 스윕의 밴드(BandX/BandY)인지
// (band_layout.hpp). 밴드를 직접 읽으면 scatter_x/y 가 필요 없다. 상수물성 판은 Field 고정.

void build_x_sweep_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, int own_axis, real_t Cmu,
                            real_t dt, SourceLayout source_layout = SourceLayout::Field);

void build_y_sweep_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, int own_axis, real_t Cmu,
                            real_t dt, SourceLayout source_layout = SourceLayout::Field);

void build_z_sweep_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, int own_axis, real_t Cmu,
                            real_t dt, SourceLayout source_layout = SourceLayout::Field);

// ── 상수물성 판 (plan §18-1) — 물성 배열 없이 UnitArray 로 같은 수식을 돈다.
//    viscosity 는 스칼라 ν (= Cmu·μ̂ 에서 μ̂≡1). 결과는 "가변 + 1 배열" 과
//    비트 일치해야 한다 (검증 게이트).
void build_x_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub, int own_axis,
                                     real_t viscosity, real_t dt);
void build_y_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub, int own_axis,
                                     real_t viscosity, real_t dt);
void build_z_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub, int own_axis,
                                     real_t viscosity, real_t dt);

}  // namespace mpmstd::equation::momentum
