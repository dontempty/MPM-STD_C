#pragma once
// 공간 발달(spatially developing) 경계층 통계 — 스팬 평균 × 시간 누적을 (x, z) 2D 로 쌓고
// 적분량 C_f(x), δ99, δ*, θ, H, Re_θ, Re_τ 와 스테이션(목표 Re_θ) 프로파일 U⁺·rms⁺·uw⁺ 를 쓴다.
// (docs/Design/20261006_boundary_layer_validation.md §3-5; 2026-10-06)
//
// 두 종류의 평균을 같이 든다:
//   relaxed  : 완화 시간평균 ⟨u⟩(i,k) (τ) — 상단 w = U∞ dδ*/dx 공급과 모니터용 (항상 갱신)
//   accumulated : dt 가중 누적 7 모멘트 (stats_start 이후) — 최종 통계 파일
// z 분할: 적분·프로파일은 2D 장을 z 로 Allgatherv 해 모든 랭크가 같은 전역 장을 갖고 계산한다
// (장 크기 nx×nz, 수 MB — 모니터 간격마다 한 번이라 무시 가능). npx = npy = 1 전제.

#include <string>
#include <vector>

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::post {

struct BoundaryLayerIntegrals {  // 전역 x 셀중심 i (크기 nx_global)
  // 적분량은 **국소 자유류 속도 u_edge(x)**(상단 5% 셀 평균)로 정규화한다 — 자유류가 U∞ 에서
  // 조금 벗어나도(상단 BC·질량보존) C_f·δ*·θ·δ99·Re_θ 가 일관되게 정의되도록 (2026-10-08).
  std::vector<real_t> x, cf, delta99, dstar, theta, shape, re_theta, re_tau, u_tau, u_edge;
};

class SpatialStatistics {
 public:
  SpatialStatistics(const grid::Grid& grid, const parallel::Subdomain& sub,
                    const std::vector<real_t>& z_center_global,
                    const std::vector<real_t>& z_face_global, real_t u_infinity, real_t nu,
                    real_t relaxation_time);

  // 매 스텝: relaxed 갱신 (+ accumulate 면 dt 가중 누적). 전 랭크 동시 호출.
  void sample(const field::Field& U, const field::Field& V, const field::Field& W, real_t dt,
              bool accumulate);
  // 적분량 — relaxed(true) 또는 accumulated(false) 평균에서. 전 랭크 동시 호출(집합 통신).
  BoundaryLayerIntegrals integrals(bool from_relaxed) const;
  // 상단 자유류 w(x) = U∞ dδ*/dx: δ*(x) 를 0.1~0.9 Lx 에서 2차 최소제곱 → 도함수. 로컬 x 인덱스
  // (ghost 포함) 로 반환. 음수는 0 으로 자른다.
  std::vector<real_t> free_stream_w(const BoundaryLayerIntegrals& in) const;
  // 통계 파일: section 1 = 적분량(x), section 2 = 스테이션(목표 Re_θ 에 가장 가까운 i) 프로파일.
  // 루트만 쓴다(집합 통신 포함 → 전 랭크 동시 호출).
  void write(const std::string& path, long step, double time,
             const std::vector<real_t>& station_re_theta) const;
  double accumulated_time() const { return time_sum_; }

 private:
  const grid::Grid& grid_;
  const parallel::Subdomain& sub_;
  std::vector<real_t> zc_, zf_;
  real_t u_inf_, nu_, tau_;
  int nxl_, nzl_, nz_global_;
  std::vector<real_t> relaxed_u_;        // nxl × nzl
  std::vector<real_t> sums_;             // 7 × nxl × nzl (dt 가중)
  double time_sum_ = 0;
  bool relaxed_ready_ = false;

  // 로컬 (nxl × nzl) 장 → 전역 (nxl × nz_global) 장 (z Allgatherv)
  std::vector<real_t> gather_z(const std::vector<real_t>& local) const;
  BoundaryLayerIntegrals integrals_of(const std::vector<real_t>& mean_u_global) const;
};

}  // namespace mpmstd::post
