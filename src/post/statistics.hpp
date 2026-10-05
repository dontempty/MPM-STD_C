#pragma once
// 공간 통계 — 수평(x·y) 평면 평균과 벌크 속도.
//
// 반환 프로파일은 항상 **전역 z 길이**다 (랭크 몫이 아니라) — 각 랭크가 자기
// z 구간을 채운 전역 벡터를 전체 합산(Allreduce)하므로 어느 분할에서도 같은
// 결과가 나온다. 나눗셈은 반드시 **전역** nx·ny 로 한다 (랭크 로컬 크기로
// 나눠 16배 틀렸던 channel WSS 진단 버그의 재발 방지).
// 시간 평균은 앱이 소유한다 — 여기 함수들은 "지금 이 장"의 공간 축소만 한다.

#include <vector>

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::post {

// ⟨F⟩_{xy}(z) — 크기 nz_global, 모든 랭크가 같은 전체 프로파일을 받는다.
std::vector<real_t> plane_average(const field::Field& F, const parallel::Subdomain& sub);

// ⟨F²⟩_{xy}(z) — rms 는 앱에서 sqrt(⟨F²⟩ − ⟨F⟩²).
std::vector<real_t> plane_average_square(const field::Field& F,
                                         const parallel::Subdomain& sub);

// ⟨F·G⟩_{xy}(z) — 대류 열유속 ⟨wT⟩(z) 같은 상관 프로파일 (슬롯값 그대로 곱 —
// RBC Nu 의 wt 누적이 쓰는 검증된 관례).
std::vector<real_t> plane_average_product(const field::Field& F,
                                          const field::Field& G,
                                          const parallel::Subdomain& sub);

// 체적가중 벌크 속도: (1/V)∫ u dV — u 는 셀 중심으로 2점 보간한 U (x-면 변수).
// 채널 CFR 강제와 모니터링이 쓴다.
real_t bulk_velocity(const field::Field& U, const grid::Grid& grid,
                     const parallel::Subdomain& sub);

// 전역 max|∇·u| — 투영 건강 모니터 (ghost 가 최신이어야 한다).
real_t max_divergence(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub);

}  // namespace mpmstd::post
