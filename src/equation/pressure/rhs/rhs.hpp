#pragma once
// 압력 Poisson 의 우변 — 증분(van Kan) 투영에서  ∇²(δp) = ∇·u*/dt.
// += 만 제공한다 (계획 §5.5). rhs 는 호출자가 선언하고 fill(0) 후 쌓는다.
//
// 호출 전 U, V, W 의 ghost 가 최신이어야 한다 (halo 교환 + 벽 처리) —
// 셀의 +면은 이웃 인덱스(U(i+1,…))라서 마지막 내부 셀이 ghost 슬롯을 읽는다.
// 수식 출처: MPM-STD_C pressure_engine_cpu 의 compute_divergence_rhs (1:1).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::pressure {

// rhs += ∇·u*/dt.   ∇·u |_c = (U_{i+1}−U_i)/dx_i + (V_{j+1}−V_j)/dy_j + (W_{k+1}−W_k)/dz_k
void add_divergence(field::Field& rhs, const field::Field& U, const field::Field& V,
                    const field::Field& W, const grid::Grid& grid,
                    const parallel::Subdomain& sub, real_t dt);

// 증분 투영의 "예측자 반쪽": 운동량 명시 RHS 에서 누적 압력의 구배를 뺀다.
//   rhs_u −= dt·(P_c − P_{i−1})/dmx_i   (면 위치의 staggered 구배; v, w 동형)
// P 의 ghost 가 최신이어야 한다. 호출 순서: 운동량 rhs 조립 뒤, solve 전.
void add_gradient_constant_property(field::Field& rhs_u, field::Field& rhs_v, field::Field& rhs_w,
                  const field::Field& P, const grid::Grid& grid,
                  const parallel::Subdomain& sub, real_t dt);

// ── 가변 밀도(NOB) 경로 ──────────────────────────────────────────────────────
// Poisson 우변의 ExtraTerm (PaScaL_TCS §214-225; dt 배율 없음 — div/dt 에 더함):
//   rhs += ∇·[(1 − (1/ρ̂)_면)·∇p̂],   p̂ = 2차 외삽 압력 예측자 (dPhat)
// 상수밀도 Poisson 해(dP)와 사후 투영(correct_velocity_variable_property)이 짝을 이뤄
// 가변 밀도 ∇·[(1/ρ̂)∇p] = div/dt 를 반복 없이 닫는다. p̂·1/ρ̂ ghost 최신 전제.
void add_extra_term_variable_property(field::Field& rhs, const field::Field& pressure_predictor,
                             const field::Field& inverse_density,
                             const grid::Grid& grid, const parallel::Subdomain& sub);

// 증분 투영 예측자의 가변 밀도 판 (PaScaL_TCS mom.f90:331/343):
//   rhs_u −= dt·(1/ρ̂)_면·(P_c − P_{i−1})/dmx_i   (v, w 동형; 면 체적가중)
void add_gradient_variable_property(field::Field& rhs_u, field::Field& rhs_v,
                           field::Field& rhs_w, const field::Field& P,
                           const field::Field& inverse_density,
                           const grid::Grid& grid, const parallel::Subdomain& sub,
                           real_t dt);

}  // namespace mpmstd::equation::pressure
