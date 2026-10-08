#pragma once
// 가변 점성(NOB) 운동량의 **명시 증분 커플링** 안정 한계로 정하는 dt (2026-10-09).
//
// 왜 필요한가: 운동량 δ-form ADI 는 own 방향 확산만 음해이고, 전치 응력 ∂_z(μ ∂_x w) 같은
// 교차 점성항은 M12/M13/M21/M23/M31/M32 증분 커플링으로 **명시**(직전 증분으로) 처리한다
// (momentum_stencil.hpp ③). 그 항의 셀 단위 이득은 혼합 확산수
//      D_× = C_μ · μ̂ · dt / (Δ_h · Δz)        (Δ_h = min(Δx,Δy), 벽에서 Δz 최소·μ̂ 최대)
// 에 비례한다. 글리세롤은 차가운 윗벽에서 μ̂ 가 2~12 배(ΔT 20~60 K)라 윗벽 첫 셀의 D_× 가 가장
// 크고, D_× 가 임계(실측 ≈ 15; MPM-STD_C E11/E12 경계 dt 0.005~0.0075 @ h-256·z-256·ΔT 20 K)를
// 넘으면 윗벽 U행↔W면 상호 펌핑이 자라 벽 근처 속도 rms 가 수십~수백 배 튄다 (이전 코드에서
// 블록 언더릴랙스 0.5 로 눌렀던 현상). 대류 CFL 만 보던 기존 dt 는 이 한계를 모른다.
//
//   dt_visc = C_v · min_k [ Δ_h · Δz_k / (C_μ · max_xy μ̂(z_k)) ],   권장 C_v ≈ 6~8 (임계의 1/2)
//
// 열적 물성이 dt 를 정하는 경로: μ̂ = μ̂(T) 이므로 ΔT 가 커질수록(윗벽이 차가울수록) dt 가 준다.
// 상수물성(μ̂ ≡ 1)에서도 같은 식이 성립하지만 보통 CFL 이 먼저 잡는다.
#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::time_integration {

struct ViscousCouplingLimit {
  real_t dt = 0;             // 허용 dt (C_v 적용)
  real_t mixed_number = 0;   // 현재 dt_reference 에서의 최대 D_× (모니터용; dt_reference 인자)
  int k_global = -1;         // 한계를 주는 z 셀(전역)
  real_t viscosity_max = 0;  // 그 셀 평면의 μ̂ 최대
};

// viscosity = μ̂ 필드(현재 물성), Cmu = 참조 점성 계수(ν_m 무차원), coupling_number = C_v.
// dt_reference 가 주어지면 그 dt 로 D_× 를 계산해 돌려준다(모니터). 전 랭크 호출(집합 통신).
ViscousCouplingLimit compute_viscous_coupling_dt(const field::Field& viscosity,
                                                 const grid::Grid& grid,
                                                 const parallel::Subdomain& sub, real_t Cmu,
                                                 real_t coupling_number,
                                                 real_t dt_reference = 0);

}  // namespace mpmstd::time_integration
