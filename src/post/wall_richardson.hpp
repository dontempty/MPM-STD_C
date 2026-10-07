#pragma once
// 벽 열(i,j)별 국소 플럭스 Richardson 수 |Ri_f| 분류 — Ng, Ooi, Lohse & Chung (2017) JFM 825 §4–§6.
//   u_τ(x,y) = √(ν |∂u/∂z|_w),  f_w(x,y) = κ |∂T/∂z|_w   (첫 셀 ↔ ghost 의 이산 벽 플럭스 그대로 —
//   ghost 정책(value/mirror)이 평균 통계의 자(ruler)와 같은 방식으로 반영된다)
//   L = −u_τ³/(gβ f_w),  |Ri_f| = z(z⁺ = z_ref)/|L| = z_ref·ν·f_w/u_τ⁴   (자유낙하 무차원: gβ = 1)
//   low-|Ri_f| (0 < |Ri_f| ≤ threshold; 논문 0.1) = 줄무늬·높은 전단 영역 → 지시자 1, 아니면 0.
//   u_τ → 0 이면 |Ri_f| → ∞ (부력 지배, high 로 분류).
// 벽을 가진 랭크(z 분할의 첫/마지막)가 계산해 전 랭크에 Bcast 하고, 각 랭크가 weights(백엔드
// 메모리, 내부 nx_local×ny_local)에 올린다 → post/statistics 의 plane_weighted_average 가 조건평균을 만든다.
// |Ri_f| 의 PDF(논문 fig.4) 용 log10 히스토그램도 돌려준다 (이번 호출의 열 수; 시간 가중은 호출자).
#include <vector>

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::post {

struct RichardsonBins {
  double log10_min = -4.0;  // |Ri_f| = 1e-4
  double log10_max = 2.0;   // |Ri_f| = 1e2
  int count = 60;           // 0.1 decade
};

struct WallRichardsonResult {
  real_t area_fraction = 0;        // ρ = low-|Ri_f| 열 수 / (nx·ny) (전역)
  std::vector<double> histogram;   // count + 2: [0] underflow, [1..count] bins, [count+1] overflow(∞ 포함)
};

// 열별 |Ri_f| (내부 nx_local×ny_local, 행 우선 (i−h)·ny_local + (j−h), 호스트 벡터) — 벽을 가진 랭크에서만
// 의미 있다. side 0 = z− (hot), 1 = z+ (cold). 백엔드별 구현: wall_richardson_kernels.{cpp,cu}.
void local_flux_richardson(std::vector<real_t>& richardson, const field::Field& U,
                           const field::Field& T, const grid::Grid& grid,
                           const parallel::Subdomain& sub, real_t nu, real_t alpha, int side,
                           real_t z_plus_reference);

// 분류 + Bcast + weights 채움 (양 백엔드 공용 오케스트레이션, wall_richardson.cpp).
// weights 크기가 nx_local·ny_local 이 아니면 assign 한다.
WallRichardsonResult classify_wall_richardson(const field::Field& U, const field::Field& T,
                                              const grid::Grid& grid,
                                              const parallel::Subdomain& sub, real_t nu,
                                              real_t alpha, int side, real_t threshold,
                                              real_t z_plus_reference,
                                              const RichardsonBins& bins,
                                              field::Buffer& weights);

// weights(지시자 I) 의 보수 1 − I 를 other 에 (high-|Ri_f| 가중) — 백엔드별 (kernels 쌍둥이)
void complement_weights(field::Buffer& other, const field::Buffer& indicator);

}  // namespace mpmstd::post
