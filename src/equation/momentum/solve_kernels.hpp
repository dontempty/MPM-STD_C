#pragma once
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
