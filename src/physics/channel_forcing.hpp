#pragma once
// 채널 유동의 평균 유량 유지 (CFR, constant flow rate) — Filtered_TDMA
// channel_gpu 의 MASS_FLOW 강제 1:1.
//
// 두 반쪽이 한 쌍이다:
// 1) add_mean_pressure_gradient — 운동량 RHS 에 추적 중인 평균 압력 구배를
//    넣는다: rhs_u += dt·(−mean_dPdx). (주기 x 에서 평균 구배는 압력 "필드"가
//    담을 수 없으므로 별도 스칼라로 다닌다.)
// 2) correct_bulk_velocity — 운동량 풀이 직후: 벌크 속도를 목표값으로 되돌리는
//    균일 이동을 가하고, 그만큼을 mean_dPdx 에 적립한다:
//      ΔG = (Ub − target)/dt,  U += −dt·ΔG,  mean_dPdx += ΔG
//    수렴하면 이동량 → 0 이 되고 mean_dPdx 가 실제 구동 구배(−τ_w/δ)로 간다.
// mean_dPdx 는 앱(main)이 소유하는 상태다 — 모니터 값이자 재시작 대상.

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics {

void add_mean_pressure_gradient(field::Field& rhs_u, const parallel::Subdomain& sub,
                                real_t mean_dPdx, real_t dt);

// ⚠ U 의 ghost 가 최신이어야 한다 (벌크 적분이 마지막 내부 셀의 +면 = ghost
// 슬롯을 읽는다). 운동량 solve 직후라면 교환부터 — 안 하면 낡은 ghost 만큼
// 벌크가 어긋나 보정이 10⁻⁵ 수준으로 샌다 (test_momentum 에서 실측된 함정).
// 호출 뒤에도 균일 이동 때문에 ghost 가 다시 낡는다 — 재교환은 호출자 몫.
void correct_bulk_velocity(field::Field& U, const grid::Grid& grid,
                           const parallel::Subdomain& sub, real_t target_bulk,
                           real_t dt, real_t& mean_dPdx);

}  // namespace mpmstd::physics
