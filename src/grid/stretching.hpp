#pragma once
// 축 하나의 "전역" 면(face) 좌표를 만든다. 규약: faces.front() = 0,
// faces.back() = length, 크기 = n_cells + 1.
// Grid 가 이 배열에서 각 랭크의 몫을 잘라낸다 (grid.hpp).
//
// 잘못된 인자(n_cells <= 0, length <= 0, gamma <= 0)는 전부 throw.

#include <vector>

#include "common/types.hpp"

namespace mpmstd::grid {

// 등간격: faces[i] = length * i / n_cells.
std::vector<real_t> make_uniform_faces(int n_cells, real_t length);

// 양쪽 벽 집중 tanh (PaScaL_TCS 식):
//   face(ξ) = L/2 · (1 + tanh(γ·(2ξ−1)/2) / tanh(γ/2)),   ξ = i / n_cells
// γ 가 클수록 벽 근처에 셀이 몰린다 (RBC 는 γ ≈ 2.6).
// 양 끝점은 0 과 length 로 강제해 부동소수 드리프트를 막는다.
std::vector<real_t> make_tanh_faces(int n_cells, real_t length, real_t gamma);

// 한쪽(z = 0)만 조밀한 tanh — 경계층(벽 하나 + 자유류 상단) 용 (2026-10-06).
//   z(ξ) = L·[1 − tanh(γ(1−ξ))/tanh γ],  ξ = i/n.  γ 가 클수록 벽에 몰린다
//   (γ=2.5, n=128, L=40 → 첫 셀 0.016 L/40, 상단 셀 ≈ 2.5/tanh γ · L/n).
std::vector<real_t> make_tanh_wall_faces(int n_cells, real_t length, real_t gamma);

}  // namespace mpmstd::grid
