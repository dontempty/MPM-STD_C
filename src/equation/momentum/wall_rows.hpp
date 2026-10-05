#pragma once
// z 벽의 밴드 행 닫기 — **두 운동량 스킴(충실·legacy OB)이 공유**하는 유틸.
// (2026-10-05 재분류에서 system.hpp 로부터 분리 — plan §18-1 후속)
//
// * fold_z_wall_value — U·V 의 z 스윕과 W 의 상단: ghost 정책 fill_z_wall_value
//   (δghost = 0) 의 쌍 = **비대각만 제거, B 는 그대로** (벽 확산 유지).
//   ⚠ B 까지 보상하면(B−=A 또는 B+=A) 벽 물리가 바뀐다 — MPM-STD_C 에서
//   모든 Dirichlet 에 B+=A 를 적용했다가 no-slip 이 under-damped 된 실화.
// * pin_z_bottom_wall_face_row — W 의 z 스윕 전용: MAC 격자에서 하단 벽면이
//   z 스윕의 **내부 행 0** 이라(상단 면은 ghost 슬롯), 일반 행으로 풀면 가짜
//   미지수가 TDMA 를 타고 이웃 행을 오염시킨다. 항등행(A=C=0, B=1, D=0)으로
//   δW_벽면 = 0 을 풀이 **전에** 박는다 (사후 0 덮어쓰기로는 복구 불가 —
//   MPM-STD_C Report/30).

#include "common/types.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

// δghost = 0 (ZeroGhost Dirichlet) 의 fold: 하단 행 A = 0, 상단 행 C = 0. B 불변.
void fold_z_wall_value(real_t* A, real_t* B, real_t* C,
                       const parallel::Subdomain& sub, int nsys, int nrow);

// W 전용: 하단 벽면(내부 행 0)을 항등행으로 (하단 벽 소유 랭크에서만).
void pin_z_bottom_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                                const parallel::Subdomain& sub, int nsys);

}  // namespace mpmstd::equation::momentum
