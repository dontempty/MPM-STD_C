#pragma once
// 가변 점성(NOB) 증분 커플링 — 블록 하삼각(전진) + 상삼각(사후).
//
//   전진 (각 성분 스윕 **전에** 우변에 주입 — Gauss-Seidel):
//     rhs_v −= dt·M21(δU)          (dU 해 직후, dV 스윕 전)
//     rhs_w −= dt·(M31(δU) + M32(δV))   (dW 스윕 전 — 벽수직은 음해로 통과)
//   사후 (상삼각 — OB 의 block_couple 과 같은 자리, 가변 μ·1/ρ̂ 판):
//     δV −= dt·M23(δW),  δU −= dt·(M12(δW) + M13(δV))
//   벽수직 δW 는 사후 보정을 받지 않는다 (재점화 수정의 핵심 — 참조 Report/RBC).
//
// 수식은 momentum_stencil.hpp 의 couple_* 셀 함수 한 곳.

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

void couple_rhs_v_from_delta_u_variable_property(field::Field& rhs_v,
                                        const field::Field& delta_u,
                                        const field::Field& V,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub, real_t Cmu,
                                        real_t dt);

void couple_rhs_w_from_delta_uv_variable_property(field::Field& rhs_w,
                                         const field::Field& delta_u,
                                         const field::Field& delta_v,
                                         const field::Field& W,
                                         const field::Field& viscosity,
                                         const field::Field& inverse_density,
                                         const grid::Grid& grid,
                                         const parallel::Subdomain& sub, real_t Cmu,
                                         real_t dt);

void block_couple_delta_v_variable_property(field::Field& delta_v, const field::Field& delta_w,
                                   const field::Field& V,
                                   const field::Field& viscosity,
                                   const field::Field& inverse_density,
                                   const grid::Grid& grid,
                                   const parallel::Subdomain& sub, real_t Cmu,
                                   real_t dt);

void block_couple_delta_u_variable_property(field::Field& delta_u, const field::Field& delta_v,
                                   const field::Field& delta_w, const field::Field& U,
                                   const field::Field& viscosity,
                                   const field::Field& inverse_density,
                                   const grid::Grid& grid,
                                   const parallel::Subdomain& sub, real_t Cmu,
                                   real_t dt);

// ── 상수물성 판 (plan §18-1) — 물성 배열 없이 같은 수식 (viscosity = 스칼라 ν).
void couple_rhs_v_from_delta_u_constant_property(
    field::Field& rhs_v, const field::Field& delta_u, const field::Field& V,
    const grid::Grid& grid, const parallel::Subdomain& sub, real_t viscosity,
    real_t dt);
void couple_rhs_w_from_delta_uv_constant_property(
    field::Field& rhs_w, const field::Field& delta_u, const field::Field& delta_v,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    real_t viscosity, real_t dt);
void block_couple_delta_v_constant_property(field::Field& delta_v,
                                            const field::Field& delta_w,
                                            const field::Field& V,
                                            const grid::Grid& grid,
                                            const parallel::Subdomain& sub,
                                            real_t viscosity, real_t dt);
void block_couple_delta_u_constant_property(field::Field& delta_u,
                                            const field::Field& delta_v,
                                            const field::Field& delta_w,
                                            const field::Field& U,
                                            const grid::Grid& grid,
                                            const parallel::Subdomain& sub,
                                            real_t viscosity, real_t dt);

}  // namespace mpmstd::equation::momentum
