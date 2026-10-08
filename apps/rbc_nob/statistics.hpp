#pragma once
// rbc_nob 통계 — Horn, Shishkina & Wagner (2013) JFM 724 의 검증 항목을 전부 누적한다 (2026-10-09).
//
//   평면(xy)·시간(dt 가중) 평균 프로파일: T̄, ū v̄ w̄, rms(u v w T), ⟨wT⟩, ⟨μ̂⟩, ⟨κ̂⟩, 운동에너지 ½⟨u²+v²+w²⟩
//   → 논문 fig.4 (T̄(z)·곡률), fig.9(c,d)(수평/연직 rms), fig.11(에너지 풍속 프로파일)
//   벽 Nu (κ̂_wall·∂T̄/∂z, 양 벽 따로) + 체적 Nu (⟨wT⟩ 적분; 기존 정의)  → fig.15, (4.4)·(4.30)
//   θ_c (중심 온도), 열 BL 두께 λ_θ,b/t (slope thickness (4.2)(4.3)) 와 비 χ_λθ  → fig.5·6
//   점성 BL 두께 λ_u,b/t (수평 rms 의 첫 극대 (4.21)(4.22)) 와 비 χ_λu            → fig.5
//   Re_E (체적 운동에너지 기반 (4.26), ν_m 과 ν_c 두 버전)                       → fig.12·13
//   온도 PDF (체적 가중 히스토그램)                                                → fig.8
//   스파이크 감시: 벽 인접 평면의 max|U| (윗벽 펌핑 진단, 모니터)
// 재시작 시 누적은 0 부터(체크포인트에 누적기 없음). 표본 간격 stats_sample_interval(스텝; 간격 안 dt 합이 가중).
#include <string>
#include <vector>

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "physics/properties.hpp"

namespace rbc_nob {

struct StatisticsConfig {
  int nz = 0;
  double lx = 0, ly = 0, lz = 0;
  double rayleigh = 0, prandtl = 0, delta_temperature = 0;  // ΔT 는 차원 K (물성 다항식 인자)
  mpmstd::real_t Cmu = 0, Ct = 0;                            // ν_m, α_m (무차원)
  mpmstd::physics::PropertyModel model;
  std::vector<mpmstd::real_t> z_center, dz;                 // 전역
  mpmstd::real_t wall_temperature_bottom = 0.5, wall_temperature_top = -0.5;
  int sample_interval = 1;
  int pdf_bins = 100;
  std::string stats_dir;
};

class Statistics {
 public:
  Statistics(const StatisticsConfig& config, const mpmstd::parallel::Subdomain& sub);

  void accumulate(long step, double t, mpmstd::real_t dt, const mpmstd::field::Field& U,
                  const mpmstd::field::Field& V, const mpmstd::field::Field& W,
                  const mpmstd::field::Field& T, const mpmstd::physics::PropertyFields& properties,
                  const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub);

  bool started() const { return sum_dt_ > 0; }
  double sum_dt() const { return sum_dt_; }
  double nusselt_volume() const;  // 1 + ⟨wT⟩_V/Ct (기존 정의)
  double theta_c() const;         // 시간평균 중심 온도

  int write(long step, double t, bool is_root) const;  // stats + tpdf 파일, 반환 = 쓴 파일 수

  // 순간 진단(모니터용): 평면 프로파일에서 벽 Nu, 중심 온도
  void wall_nusselt(const std::vector<mpmstd::real_t>& t_bar, double& nu_bottom,
                    double& nu_top) const;
  double center_temperature(const std::vector<mpmstd::real_t>& t_bar) const;
  mpmstd::real_t kappa_wall_bottom() const { return kappa_wall_bottom_; }
  mpmstd::real_t kappa_wall_top() const { return kappa_wall_top_; }

 private:
  using real_t = mpmstd::real_t;
  void mean(const std::vector<double>& sum, std::vector<double>& out) const;

  StatisticsConfig c_;
  real_t kappa_wall_bottom_ = 1, kappa_wall_top_ = 1;  // κ̂(T_wall)
  real_t z_first_ = 0, z_last_gap_ = 0;
  int kc_low_ = 0, kc_high_ = 0;

  double sum_dt_ = 0, pending_dt_ = 0, time_begin_ = -1;
  std::vector<double> sum_t_, sum_u_, sum_v_, sum_w_, sum_uu_, sum_vv_, sum_ww_, sum_tt_, sum_wt_,
      sum_mu_, sum_kappa_;
  double sum_wt_integral_ = 0, sum_theta_c_ = 0;
  std::vector<double> histogram_;
  double volume_ = 0;
};

}  // namespace rbc_nob
