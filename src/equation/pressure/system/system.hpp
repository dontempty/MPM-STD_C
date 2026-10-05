#pragma once
// 압력 Poisson 의 시스템 계수 — 스펙트럴 x·y + 실공간 z.
//
// x, y 가 주기·균일이라 FFT 가 그 두 방향을 대각화한다. 남는 것은 (kx, ky)
// 모드마다 z 삼중대각 하나:
//
//   a_k·φ̂_{k−1} + b_k·φ̂_k + c_k·φ̂_{k+1} = rhŝ_k
//   a_k = 1/(dz_k·dmz_k),  c_k = 1/(dz_k·dmz_{k+1}),  b_k = −(a_k + c_k) − λ
//
// λ 는 연속 파수 k² 가 아니라 **2차 중심차분의 고유값** (modified wavenumber):
//   λ_x[m] = (1 − cos(2π·m/Nx)) · 2/dx²     (균일격자 전제 — Solver 가 검사)
// 이것을 써야 "FFT 로 푼 해"가 "차분 라플라시안의 해"와 정확히 일치해
// 투영 후 ∇·u 가 기계 0 으로 떨어진다.
//
// z 벽(Neumann, ∂φ/∂z = 0)은 조립 시 접는다: 하단 벽 행 a = 0, 상단 벽 행 c = 0.
// 이는 conventions.md §6 의 zero_gradient ↔ B += A 쌍과 동일한 식이다
// (b = −(a+c)−λ 에 a 를 0 으로 두는 것 = B 에 +a 보상을 미리 반영한 꼴).
// ghost 쪽 짝은 solve 가 끝에 적용하는 fill_z_wall_zero_gradient.
//
// (0,0) 모드는 λ = 0 + 양쪽 Neumann 이라 특이(널공간 = 상수) — 전역 z 첫 행을
// 항등행으로 핀 (D 는 solve 가 0 으로). 상수 게이지는 해 뒤 전역 평균 제거.
//
// 밴드 레이아웃은 TDMA 계약 그대로 [sys + row·nsys]:
//   row = k − h (로컬),  nsys = 2·(Nx/2+1)·Ny,
//   sys = 2·(mx·Ny + jy) + {0: 실수부, 1: 허수부}   — 실·허가 같은 계수를 쓰므로
//   한 번의 분산 TDMA 호출로 두 성분을 같이 푼다.

#include <vector>

#include "common/types.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::pressure {

// 2차 중심차분 고유값 표. 크기 Nx/2+1 (r2c 절반 스펙트럼) / Ny.
std::vector<real_t> make_lambda_x(const grid::Grid& grid, const parallel::Subdomain& sub);
std::vector<real_t> make_lambda_y(const grid::Grid& grid, const parallel::Subdomain& sub);

// 모든 (kx, ky) 모드의 z 밴드를 채운다 (위 주석의 식·레이아웃·벽 접기·핀).
// λ 표는 그 빌드의 백엔드 메모리 포인터다 (CPU=호스트, GPU=디바이스 —
// pressure::Solver 가 field::Buffer 로 소유).
void build_z_modes_system(real_t* A, real_t* B, real_t* C, const grid::Grid& grid,
                          const parallel::Subdomain& sub, const real_t* lambda_x,
                          int n_mode_x, const real_t* lambda_y, int n_mode_y);

}  // namespace mpmstd::equation::pressure
