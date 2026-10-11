#pragma once
// 공간 발달 경계층의 **수동 스칼라(온도) 통계** — SpatialStatistics(속도)의 짝 (2026-10-11,
// 열 경계층 검증: Li, Schlatter, Brandt & Henningson 2009 IJHFF 30, 916).
//   스팬 평균 × dt 가중 시간 누적을 (x, z) 2D 로 쌓아
//   section 1: x 마다 St, θ_w, (∂θ/∂z)_w, δ_θ99, θ_τ, 2St/C_f
//   section 2: 스테이션(속도 통계와 같은 i) 프로파일 θ⁺, θ_rms⁺, ⟨u′θ′⟩⁺, ⟨w′θ′⟩⁺ (+ 원시 평균)
// 벽 조건 두 가지(Li 2009 eq. 10–11):
//   isoscalar : θ_w = wall_value (보통 0), θ∞ = 1 — St = α (∂θ/∂z)_w / (u_e (θ∞ − θ_w))
//   isoflux   : (∂θ/∂z)_w = wall_gradient (보통 1), θ∞ = 1 — θ_w 는 2점 외삽; St = α g_w /(u_e (θ∞ − θ_w))
// θ_τ = α (∂θ/∂z)_w / u_τ (마찰 스칼라 q_w/(ρ c_p u_τ)), θ⁺ = (θ − θ_w)/θ_τ.
// z 분할: 2D 장을 z 로 Allgatherv (SpatialStatistics 와 동일). npx = npy = 1 전제.
#include <string>
#include <vector>

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "post/spatial_statistics.hpp"

namespace mpmstd::post {

enum class ScalarWall { Isoscalar, Isoflux };

struct ScalarIntegrals {  // 전역 x 셀중심 i
  std::vector<real_t> x, stanton, theta_wall, gradient_wall, delta99_theta, theta_tau,
      reynolds_analogy;  // 2St/C_f
};

class ScalarSpatialStatistics {
 public:
  ScalarSpatialStatistics(const grid::Grid& grid, const parallel::Subdomain& sub,
                          const std::vector<real_t>& z_center_global,
                          const std::vector<real_t>& z_face_global, real_t nu, real_t prandtl,
                          ScalarWall wall, real_t wall_value, real_t wall_gradient,
                          real_t theta_infinity);

  // 매 스텝(accumulate 면 dt 가중 누적). 전 랭크 동시 호출.
  void sample(const field::Field& U, const field::Field& W, const field::Field& T, real_t dt,
              bool accumulate);
  // 적분량: 속도 통계의 BoundaryLayerIntegrals(같은 창)로 u_e·u_τ·C_f 를 받는다.
  ScalarIntegrals integrals(const BoundaryLayerIntegrals& velocity) const;
  // 통계 파일 — 루트만 쓴다(집합 통신 포함 → 전 랭크 동시 호출). 스테이션 i 는 속도 통계와 같은
  // 규칙(목표 Re_θ 에 가장 가까운 i)으로 고른다.
  void write(const std::string& path, long step, double time,
             const std::vector<real_t>& station_re_theta,
             const BoundaryLayerIntegrals& velocity) const;
  double accumulated_time() const { return time_sum_; }

 private:
  const grid::Grid& grid_;
  const parallel::Subdomain& sub_;
  std::vector<real_t> zc_, zf_;
  real_t nu_, prandtl_, alpha_;
  ScalarWall wall_;
  real_t wall_value_, wall_gradient_, theta_inf_;
  int nxl_, nzl_, nz_global_;
  std::vector<real_t> sums_;  // 6 × nxl × nzl (dt 가중): θ θ² u_c w_c u_cθ w_cθ
  double time_sum_ = 0;

  std::vector<real_t> gather_z(const std::vector<real_t>& local) const;
  // 벽값·벽 구배 (2점 편측) — 평균 θ 프로파일 한 기둥에서
  void wall_state(const real_t* theta, real_t& theta_wall, real_t& gradient_wall) const;
  ScalarIntegrals integrals_of(const std::vector<real_t>& mean_theta_global,
                               const BoundaryLayerIntegrals& velocity) const;
};

}  // namespace mpmstd::post
