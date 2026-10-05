#pragma once
// 에너지 방정식의 음해(implicit) 시스템 계수 — 축별 (I − ½·dt·α·L_d) 밴드.
//
//   L_d φ_i = ((φ_{i+1}−φ_i)/dmx_{i+1} − (φ_i−φ_{i−1})/dmx_i) / dx_i
//   A_i = −½dtα/(dx_i·dmx_i)      (아래 비대각)
//   C_i = −½dtα/(dx_i·dmx_{i+1})  (위 비대각)          dx = cell_width
//   B_i = 1 − A_i − C_i           (대각)               dmx = center_distance
//
// 밴드·D 레이아웃은 TDMA 계약과 동일한 [sys + row·nsys] (sys 가 빠름).
// sys 배치 — 스윕 축이 row, 나머지 두 축이 sys:
//   x-스윕: row = i−h,  sys = (j−h)·nz_local + (k−h),  nsys = ny_local·nz_local
//   y-스윕: row = j−h,  sys = (i−h)·nz_local + (k−h),  nsys = nx_local·nz_local
//   z-스윕: row = k−h,  sys = (i−h)·ny_local + (j−h),  nsys = nx_local·ny_local
// D 에는 source(ghost 포함 전체 모양 배열)의 내부가 같은 배치로 옮겨 담긴다.
// 계수는 축 1차원 메트릭만 쓰므로 한 row 의 모든 sys 가 같은 (A,B,C)다.
// (D13: 축 매개변수화 금지 — 세 함수를 따로 쓴다. 공유는 스칼라 수식 하나뿐.)

#include "common/types.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::scalar {

void build_x_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    real_t alpha, real_t dt);
void build_y_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    real_t alpha, real_t dt);
void build_z_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    real_t alpha, real_t dt);

// z 벽 행의 fold — T 의 ghost 정책 fill_z_wall_mirror 와 **쌍**이다
// (docs/conventions.md §6: 쌍이 어긋나면 벽의 물리가 바뀐다).
//   mirror: ghost = 2v − f₁  →  δghost = −δf₁  →  하단 행 B −= A, 상단 행 B −= C
//   (비대각은 0 으로 제거, D 는 불변)
// ★ "D 불변"은 **벽값 상수** 가정이다 — 벽 온도가 시간에 따라 변하면(v(t))
//   벽 행 D 에 −2Δv·A 항을 추가해야 한다 (conventions.md §6 체크리스트).
//   현재 케이스(RBC ±0.5, DHVC)는 전부 상수.
// 벽을 소유한 랭크에서만 동작한다 (neighbor == MPI_PROC_NULL 판정 내장).
void fold_z_wall_mirror(real_t* A, real_t* B, real_t* C,
                        const parallel::Subdomain& sub, int nsys, int nrow);

}  // namespace mpmstd::equation::scalar
