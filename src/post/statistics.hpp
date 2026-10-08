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
#include "field/buffer.hpp"
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
// 스팬(y) 평균 — 셀중심 보간 속도의 7개 모멘트 ⟨uc⟩⟨vc⟩⟨wc⟩⟨uc²⟩⟨vc²⟩⟨wc²⟩⟨uc·wc⟩ 를
// 로컬 (i,k) 내부 격자마다 (uc = ½(U_i + U_{i+1}) 등). out 크기 = 7 × nx_local × nz_local,
// 색인 m·n + (i_local·nz_local + k_local). 공간 발달 통계(post/spatial_statistics)의 재료.
// npy = 1 전제 (y 가 랭크 안에서 완결) — 어기면 throw.
void span_average_center_moments(const field::Field& U, const field::Field& V,
                                 const field::Field& W, std::vector<real_t>& out,
                                 const parallel::Subdomain& sub);

real_t bulk_velocity(const field::Field& U, const grid::Grid& grid,
                     const parallel::Subdomain& sub);

// 전역 max|∇·u| — 투영 건강 모니터 (ghost 가 최신이어야 한다).
real_t max_divergence(const field::Field& U, const field::Field& V,
                      const field::Field& W, const grid::Grid& grid,
                      const parallel::Subdomain& sub);

// ── 열(x,y) 가중 평면 평균 — 조건평균(국소 |Ri_f| 분류, post/wall_richardson) 용 (2026-10-07) ──
//   weights: 내부 nx_local×ny_local (행 우선 (i−h)·ny_local + (j−h)), 백엔드 메모리(field::Buffer),
//   0/1 지시자 또는 임의 가중. 반환 = Σ_ij w·F / Σ_ij w (둘 다 전역 Allreduce); Σw = 0 이면 전부 0.
//   plane_average 와 같은 분할 규칙(어느 분할에서도 같은 전역 프로파일).
std::vector<real_t> plane_weighted_average(const field::Field& F, const field::Buffer& weights,
                                           const parallel::Subdomain& sub);
std::vector<real_t> plane_weighted_average_square(const field::Field& F,
                                                  const field::Buffer& weights,
                                                  const parallel::Subdomain& sub);
std::vector<real_t> plane_weighted_average_product(const field::Field& F, const field::Field& G,
                                                   const field::Buffer& weights,
                                                   const parallel::Subdomain& sub);

// 평면 최대 max_{xy} F(z) (absolute=true 면 |F|) — 크기 nz_global, 전 랭크 동일 (Allreduce MAX).
// 용도: 벽 근처 폭주 감시(|U| 최대), 가변 점성 dt 기준(μ̂ 평면 최대; time/time_step).
std::vector<real_t> plane_maximum(const field::Field& F, const parallel::Subdomain& sub,
                                  bool absolute);

// 체적 가중 히스토그램 — F 의 내부 셀 값을 [lower, upper) 를 count 칸으로 나눠 셀 체적(dx·dy·dz)을
// 더한다. 반환 크기 count + 2: [0] 아래 넘침, [1..count] 칸, [count+1] 위 넘침. 전 랭크 동일(Allreduce).
// 용도: 온도 PDF (Horn 2013 fig.8). 정규화(÷총 체적·칸폭)는 호출자.
std::vector<double> volume_histogram(const field::Field& F, const grid::Grid& grid,
                                     const parallel::Subdomain& sub, real_t lower, real_t upper,
                                     int count);

}  // namespace mpmstd::post
