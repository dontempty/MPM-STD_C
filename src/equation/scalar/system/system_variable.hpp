#pragma once
// 가변-κ̂ (NOB) 에너지 방정식의 스윕별 시스템 조립 — 융합 델타형.
//
// OB(상수 확산) 모듈과 달리 rhs 를 따로 조립하지 않는다: 첫 스윕(x)이 밴드와
// 우변(R − ΣM·θⁿ)을 한 번에 팩하고, 나중 스윕(y, z)은 이전 스윕의 증분을
// D 로 통과시키며 자기 축 밴드만 만든다 (PaScaL_TCS 융합 구조 1:1).
//
// z 벽 닫기는 OB 의 fold_z_wall_mirror 가 아니라 **조립 안의 마스크**가 맡는다
// (scalar_variable_stencil.hpp 머리말) — build_z_system_variable 뒤에 별도
// fold 호출이 없어야 한다.
//
// 레이아웃·순서는 OB system.hpp 와 동일: [sys + row·nsys], 루프 i→j→k.

#include "common/types.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "physics/properties.hpp"

namespace mpmstd::equation::scalar {

// 첫 스윕 (x, 주기): 밴드 + 우변 팩 (R 은 세 축 전부에서 계산)
void build_x_system_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* T, const real_t* U, const real_t* V,
                             const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             real_t Ct, real_t dt);

// 나중 스윕 (y 주기 / z 벽): 자기 축 밴드만, D = 이전 스윕 증분(source)
void build_y_system_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             real_t Ct, real_t dt);

void build_z_system_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             real_t Ct, real_t dt);

}  // namespace mpmstd::equation::scalar
