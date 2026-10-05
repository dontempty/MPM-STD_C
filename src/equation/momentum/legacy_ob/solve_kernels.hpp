#pragma once
// ⚠ legacy OB 경로 — 간이 스킴(보존형 RHS + 짝 없는 이류형 음해항). 블록
// 커플링의 평균이 0 이 아니어서 운동량이 dt 에 비례해 샌다(2026-10-04 수지
// 계측). **새 코드에서 쓰지 말 것** — 충실 경로(상위 디렉터리)의
// *_variable_property / *_constant_property 를 쓴다. rbc(OB) 앱 하위호환으로만
// 남아 있고, rbc 전환(Nu 회귀 재검증) 후 디렉터리째 삭제 예정.
// momentum::solve 의 백엔드 커널 묶음 (내부 전용) — CPU(solve_kernels.cpp)와
// GPU(solve_kernels.cu)가 같은 선언을 구현하고, 빌드가 한쪽 TU 만 담는다.
// 커플링 수식은 momentum_stencil.hpp 가 정본이다.

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

// (산포·증분 갱신은 equation/sweep_kernels.hpp 공용을 쓴다)

// 블록 커플링 — δV ← δW (M23), δU ← 보정된 δV + δW (M12+M13).
void block_couple_delta_v(field::Field& dV, const field::Field& dW,
                          const field::Field& V, const grid::Grid& grid,
                          const parallel::Subdomain& sub, real_t dt);
void block_couple_delta_u(field::Field& dU, const field::Field& dV,
                          const field::Field& dW, const field::Field& U,
                          const grid::Grid& grid, const parallel::Subdomain& sub,
                          real_t dt);

}  // namespace mpmstd::equation::momentum
