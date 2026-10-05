#pragma once
// ⚠ legacy OB 경로 — 간이 스킴(보존형 RHS + 짝 없는 이류형 음해항). 블록
// 커플링의 평균이 0 이 아니어서 운동량이 dt 에 비례해 샌다(2026-10-04 수지
// 계측). **새 코드에서 쓰지 말 것** — 충실 경로(상위 디렉터리)의
// *_variable_property / *_constant_property 를 쓴다. rbc(OB) 앱 하위호환으로만
// 남아 있고, rbc 전환(Nu 회귀 재검증) 후 디렉터리째 삭제 예정.
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
// z 벽 처리(fold/핀)는 wall_rows.hpp 로 분리 — 충실·legacy 두 스킴이 공유한다.

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



}  // namespace mpmstd::equation::momentum
