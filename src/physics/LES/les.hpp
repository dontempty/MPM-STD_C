#pragma once
// LES 와점성 모델 공통 — 모델은 디렉터리로 관리한다 (사용자 결정 2026-10-06):
//   physics/LES/constant_vreman/   Vreman 2004, 참조 Fortran MPM-STD `Const-Vreman` 1:1
//   physics/LES/WALE/              Nicoud & Ducros 1999 (다음)
// 모든 모델의 계약은 같다: uⁿ 에서 ν_t(셀중심)를 구해 **μ̂ = 1 + ν_t/ν** 를 쓴다 (운동량의
// ν_eff = ν·μ̂ — 가변물성 커널의 viscosity 자리; 에너지는 κ̂ = 1 + ν_t/(Pr_t·α)).
// 호출자가 뒤에 halo 교환 + 벽 ghost μ̂ = 1 (ν_t = 0 → 벽면 점성 = ν) 을 맞춘다.
// 시스템을 푸는 것이 아니므로 이름은 calculate_* (solve_* 아님).
// 설계: docs/Design/20261006_les_design.md
#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics::les {

// 벽 ghost μ̂ = 1 + 주기 halo 교환 — 모든 모델 뒤에 공통으로 (CPU/GPU 공용)
void synchronize_viscosity_ratio(field::Field& mu_hat, const parallel::Subdomain& sub);

}  // namespace mpmstd::physics::les
