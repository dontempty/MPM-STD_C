#pragma once
// 운동량 방정식의 음해(implicit) 시스템 계수 — 축별 밴드
//
//   (I − ½·dt·ν·L_d − ¼·dt·C_d)   (각 스윕; MPM-STD_C build_adi_bands 1:1)
//
// * 확산 L_d: CN 가중 ½. z 스윕은 성분의 z 슬롯에 따라 메트릭이 갈린다
//   (셀 중심 U,V vs z-면 W — rhs/rhs.hpp 의 라플라시안과 같은 구분).
// * 대류 C_d: 계수 ¼ 의 준음해 자기이송 — 이송속도는 스윕 방향의 면 속도
//   (x 스윕 = U, y = V, z = W; 모든 성분이 같은 이송장을 쓴다 — 검증된 단순화).
//   면 속도는 2점 평균, 비대각 ∓¼dt·u_면/중심간거리.
// * 밴드·D 레이아웃·sys 배치는 scalar/system.hpp 와 동일 ([sys + row·nsys]).
//
// z 벽 처리 (conventions.md §6 쌍):
// * fold_z_wall_value — U, V 의 z 스윕과 W 의 상단: ghost 정책 fill_z_wall_value
//   (δghost = 0) 의 쌍 = **비대각만 제거, B 는 그대로** (벽 확산 유지).
//   ⚠ B 까지 보상하면(B−=A 또는 B+=A) 벽 물리가 바뀐다 — MPM-STD_C 에서
//   모든 Dirichlet 에 B+=A 를 적용했다가 no-slip 이 under-damped 된 실화.
// * pin_z_bottom_wall_face_row — W 의 z 스윕 전용: MAC 격자에서 하단 벽면이
//   z 스윕의 **내부 행 0** 이라(상단 면은 ghost 슬롯), 일반 행으로 풀면 가짜
//   미지수가 TDMA 를 타고 이웃 행을 오염시킨다. 항등행(A=C=0, B=1, D=0)으로
//   δW_벽면 = 0 을 풀이 **전에** 박는다 (사후 0 덮어쓰기로는 복구 불가 —
//   MPM-STD_C Report/30).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

void build_x_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const field::Field& U_advect, const grid::Grid& grid,
                    const parallel::Subdomain& sub, real_t viscosity, real_t dt);
void build_y_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const field::Field& V_advect, const grid::Grid& grid,
                    const parallel::Subdomain& sub, real_t viscosity, real_t dt);
void build_z_system_cell(real_t* A, real_t* B, real_t* C, real_t* D,
                         const real_t* source, const field::Field& W_advect,
                         const grid::Grid& grid, const parallel::Subdomain& sub,
                         real_t viscosity, real_t dt);
void build_z_system_z_face(real_t* A, real_t* B, real_t* C, real_t* D,
                           const real_t* source, const field::Field& W_advect,
                           const grid::Grid& grid, const parallel::Subdomain& sub,
                           real_t viscosity, real_t dt);

// δghost = 0 (ZeroGhost Dirichlet) 의 fold: 하단 행 A = 0, 상단 행 C = 0. B 불변.
void fold_z_wall_value(real_t* A, real_t* B, real_t* C,
                       const parallel::Subdomain& sub, int nsys, int nrow);

// W 전용: 하단 벽면(내부 행 0)을 항등행으로 (하단 벽 소유 랭크에서만).
void pin_z_bottom_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                                const parallel::Subdomain& sub, int nsys);

}  // namespace mpmstd::equation::momentum
