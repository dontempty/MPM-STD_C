#pragma once
// DHVC 통계 누적·출력 — Ng, Ooi, Lohse & Chung (2017) JFM 825 의 통계량을 전부 검증할 수 있게 (2026-10-07).
//
//   1) 평면 평균 프로파일 (dt 가중): U V W T, ⟨u²⟩⟨v²⟩⟨w²⟩⟨T²⟩, ⟨uw⟩⟨uT⟩⟨wT⟩ (슬롯곱: U x-면·W z-면·T 중심)
//      → rms·플럭스·열/운동량 수지, Re_δ* = U δ*/ν, 벽 Nu·u_τ·θ_τ, Re = U_rms(δ_max)·H/ν, 평균량의 |Ri_f|
//      (논문 fig.7a,b·8·10 검은 점)                                         → dhvc_stats_<step>.dat
//   2) 조건평균 — 벽 열(i,j)별 국소 |Ri_f| 로 low(≤ threshold)/high 분류(양 벽 각각, post/wall_richardson):
//      ū T̄ ⟨u²⟩⟨T²⟩⟨uw⟩⟨wT⟩ 의 low/high 조건 평면평균 + 면적비 ρ + |Ri_f| 히스토그램
//      (fig.4·5·6·7c–f·8·9·10 파란/빨간 점; 논문 (6.2)–(6.4), (7.1))         → dhvc_conditional_<step>.dat,
//                                                                              dhvc_richardson_pdf_<step>.dat
//   3) u 의 1차원 에너지 스펙트럼 E(k_x; z)·E(k_y; z) (fig.11; premultiplied·u_τ² 정규화는 노트북)
//                                                                            → dhvc_spectra_<step>.dat
//   표본 간격(스텝): sample_interval(1), conditional_interval(2; 0 = sample_interval), spectra_interval(3; 0 = 끔).
//   간격 사이의 dt 를 합쳐 그 표본의 가중으로 쓴다(dt 가중 평균 유지). 1e8 급에선 10 정도가 비용 상 적당.
//   누적기는 체크포인트에 들어가지 않는다 — 재시작하면 0 부터.
#include <memory>
#include <string>
#include <vector>

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "post/spectra.hpp"
#include "post/wall_richardson.hpp"

namespace dhvc {

struct StatisticsConfig {
  int nz = 0;                         // 전역 z 셀 수
  double lx = 0, ly = 0, lz = 0;
  double rayleigh = 0, prandtl = 0;
  mpmstd::real_t nu = 0, alpha = 0;
  bool mirror_wall = false;           // 속도 ghost 정책 → u_τ 자(ruler)
  std::vector<mpmstd::real_t> z_center;  // 전역 셀 중심 (nz)
  mpmstd::real_t ghost_gap_hot = 0;   // z₁ − z_ghost (value ghost 자)
  mpmstd::real_t ghost_gap_cold = 0;  // z_ghost − z_nz
  mpmstd::real_t wall_temperature_hot = 0.5, wall_temperature_cold = -0.5;
  int sample_interval = 1;
  int conditional_interval = 0;
  int spectra_interval = 0;
  double richardson_threshold = 0.1;  // 논문 0.1
  double richardson_z_plus = 15;      // 논문 z⁺ = 15
  std::string stats_dir;
};

class Statistics {
 public:
  Statistics(const StatisticsConfig& config, const mpmstd::parallel::Subdomain& sub);

  // 통계 시작 이후 매 스텝 호출 (step 은 갱신 뒤 값, t 는 갱신 뒤 시각). 표본 간격 판단은 내부.
  void accumulate(long step, double t, mpmstd::real_t dt, mpmstd::real_t dpdx_applied,
                  const mpmstd::field::Field& U, const mpmstd::field::Field& V,
                  const mpmstd::field::Field& W, const mpmstd::field::Field& T,
                  const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub);

  bool started() const { return sum_dt_ > 0; }
  double sum_dt() const { return sum_dt_; }
  double time_begin() const { return time_begin_; }
  mpmstd::real_t running_re_delta() const;

  // 파일 4종을 stats_dir 에 (루트만). 반환: 쓴 파일 수.
  int write(long step, double t, bool is_root) const;

  // 순간 벽 진단(모니터용) — 면평균 프로파일에서. 열은 T ghost 가 거울이라 (T_wall − T₁)/z₁,
  // 속도는 ghost 정책대로(mirror: 벽면↔첫 중심, value: ghost 중심↔첫 중심).
  void wall_nusselt(const std::vector<mpmstd::real_t>& t_bar, double& nu_hot,
                    double& nu_cold) const;
  void wall_friction_velocity(const std::vector<mpmstd::real_t>& u_bar, double& ut_hot,
                              double& ut_cold) const;

 private:
  using real_t = mpmstd::real_t;
  void sample_profiles(real_t weight, const mpmstd::field::Field& U, const mpmstd::field::Field& V,
                       const mpmstd::field::Field& W, const mpmstd::field::Field& T,
                       const mpmstd::parallel::Subdomain& sub);
  void sample_conditional(real_t weight, const mpmstd::field::Field& U,
                          const mpmstd::field::Field& W, const mpmstd::field::Field& T,
                          const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub);
  void sample_spectra(real_t weight, const mpmstd::field::Field& U,
                      const mpmstd::parallel::Subdomain& sub);
  void mean_profile(const std::vector<double>& sum, std::vector<double>& out) const;
  int write_profiles(long step, double t) const;
  int write_conditional(long step, double t) const;
  int write_richardson_pdf(long step, double t) const;
  int write_spectra(long step, double t) const;

  StatisticsConfig c_;
  real_t z_first_ = 0, z_last_gap_ = 0, ruler_hot_ = 0, ruler_cold_ = 0;

  // 1) 평면 평균
  double sum_dt_ = 0, pending_dt_ = 0, time_begin_ = -1;
  double sum_dpdx_ = 0;
  std::vector<double> sum_u_, sum_v_, sum_w_, sum_t_, sum_uu_, sum_vv_, sum_ww_, sum_tt_, sum_uw_,
      sum_ut_, sum_wt_;

  // 2) 조건평균 — [side 0 hot, 1 cold][class 0 low, 1 high][quantity 0..5 = u T uu tt uw wT]
  double cond_sum_dt_ = 0, pending_cond_dt_ = 0;
  double rho_sum_[2] = {0, 0};
  mpmstd::post::RichardsonBins bins_;
  std::vector<double> histogram_[2];
  std::vector<double> cond_sum_[2][2][6];
  mpmstd::field::Buffer weight_low_[2], weight_high_[2];

  // 3) 스펙트럼
  double spec_sum_dt_ = 0, pending_spec_dt_ = 0;
  std::unique_ptr<mpmstd::post::LineSpectrumPlan> spectrum_plan_;
  mpmstd::post::LineSpectra spectra_sum_;
};

}  // namespace dhvc
