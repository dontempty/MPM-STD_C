#include "dhvc/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "post/statistics.hpp"

namespace dhvc {

using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;

namespace {

// u_max 와 δ* = ∫₀^{z(u_max)} (1 − U/u_max) dz (참조 compute_re_delta_star 1:1; 벽 z=0 에서 1 로 시작)
void maximum_and_displacement_thickness(const std::vector<double>& u_profile,
                                        const std::vector<mpmstd::real_t>& z_center,
                                        double& u_max, double& dstar, int& k_max) {
  const int nz = static_cast<int>(u_profile.size());
  k_max = 0;
  u_max = u_profile[0];
  for (int k = 0; k < nz; ++k)
    if (u_profile[k] > u_max) {
      u_max = u_profile[k];
      k_max = k;
    }
  dstar = 0.0;
  if (u_max <= 0.0) return;
  double z_prev = 0.0, f_prev = 1.0;
  for (int k = 0; k <= k_max; ++k) {
    const double z = z_center[k];
    const double f = 1.0 - u_profile[k] / u_max;
    dstar += 0.5 * (f_prev + f) * (z - z_prev);
    z_prev = z;
    f_prev = f;
  }
}

double rms_of(const double square_mean, const double mean) {
  return std::sqrt(std::max(0.0, square_mean - mean * mean));
}

}  // namespace

Statistics::Statistics(const StatisticsConfig& config, const Subdomain& sub) : c_(config) {
  if (c_.nz <= 0 || static_cast<int>(c_.z_center.size()) != c_.nz)
    throw std::invalid_argument("dhvc::Statistics: z_center size must equal nz");
  if (c_.sample_interval <= 0) throw std::invalid_argument("dhvc::Statistics: sample_interval > 0");
  z_first_ = c_.z_center[0];
  z_last_gap_ = static_cast<real_t>(c_.lz) - c_.z_center[c_.nz - 1];
  ruler_hot_ = c_.mirror_wall ? z_first_ : c_.ghost_gap_hot;
  ruler_cold_ = c_.mirror_wall ? z_last_gap_ : c_.ghost_gap_cold;
  for (std::vector<double>* s : {&sum_u_, &sum_v_, &sum_w_, &sum_t_, &sum_uu_, &sum_vv_, &sum_ww_,
                                 &sum_tt_, &sum_uw_, &sum_ut_, &sum_wt_})
    s->assign(c_.nz, 0.0);
  for (int side = 0; side < 2; ++side) {
    histogram_[side].assign(bins_.count + 2, 0.0);
    for (int cls = 0; cls < 2; ++cls)
      for (int q = 0; q < 6; ++q) cond_sum_[side][cls][q].assign(c_.nz, 0.0);
  }
  if (c_.spectra_interval > 0) spectrum_plan_ = std::make_unique<mpmstd::post::LineSpectrumPlan>(sub);
}

void Statistics::wall_nusselt(const std::vector<real_t>& t_bar, double& nu_hot,
                              double& nu_cold) const {
  const double delta_t = c_.wall_temperature_hot - c_.wall_temperature_cold;
  nu_hot = double(c_.wall_temperature_hot - t_bar[0]) / z_first_ * c_.lz / delta_t;
  nu_cold = double(t_bar[c_.nz - 1] - c_.wall_temperature_cold) / z_last_gap_ * c_.lz / delta_t;
}

void Statistics::wall_friction_velocity(const std::vector<real_t>& u_bar, double& ut_hot,
                                        double& ut_cold) const {
  ut_hot = std::sqrt(double(c_.nu) * std::abs(double(u_bar[0])) / ruler_hot_);
  ut_cold = std::sqrt(double(c_.nu) * std::abs(double(u_bar[c_.nz - 1])) / ruler_cold_);
}

void Statistics::mean_profile(const std::vector<double>& sum, std::vector<double>& out) const {
  out.resize(c_.nz);
  for (int q = 0; q < c_.nz; ++q) out[q] = sum_dt_ > 0 ? sum[q] / sum_dt_ : 0.0;
}

mpmstd::real_t Statistics::running_re_delta() const {
  if (sum_dt_ <= 0) return 0;
  std::vector<double> u_mean;
  mean_profile(sum_u_, u_mean);
  double u_max = 0, dstar = 0;
  int k_max = 0;
  maximum_and_displacement_thickness(u_mean, c_.z_center, u_max, dstar, k_max);
  return u_max > 0 ? static_cast<real_t>(u_max * dstar / c_.nu) : real_t(0);
}

void Statistics::accumulate(const long step, const double t, const real_t dt,
                            const real_t dpdx_applied, const Field& U, const Field& V,
                            const Field& W, const Field& T, const mpmstd::grid::Grid& grid,
                            const Subdomain& sub) {
  if (time_begin_ < 0) time_begin_ = t - dt;
  pending_dt_ += dt;
  pending_cond_dt_ += dt;
  pending_spec_dt_ += dt;
  sum_dpdx_ += double(dpdx_applied) * dt;  // 매 스텝 (가벼움) — 창 평균 등가 ∂P/∂x
  if (step % c_.sample_interval == 0) {
    sample_profiles(static_cast<real_t>(pending_dt_), U, V, W, T, sub);
    sum_dt_ += pending_dt_;
    pending_dt_ = 0;
  }
  const int conditional_interval =
      c_.conditional_interval > 0 ? c_.conditional_interval : c_.sample_interval;
  if (step % conditional_interval == 0) {
    sample_conditional(static_cast<real_t>(pending_cond_dt_), U, W, T, grid, sub);
    cond_sum_dt_ += pending_cond_dt_;
    pending_cond_dt_ = 0;
  }
  if (c_.spectra_interval > 0 && step % c_.spectra_interval == 0) {
    sample_spectra(static_cast<real_t>(pending_spec_dt_), U, sub);
    spec_sum_dt_ += pending_spec_dt_;
    pending_spec_dt_ = 0;
  }
}

void Statistics::sample_profiles(const real_t weight, const Field& U, const Field& V,
                                 const Field& W, const Field& T, const Subdomain& sub) {
  namespace post = mpmstd::post;
  const auto u_bar = post::plane_average(U, sub);
  const auto v_bar = post::plane_average(V, sub);
  const auto w_bar = post::plane_average(W, sub);
  const auto t_bar = post::plane_average(T, sub);
  const auto uu = post::plane_average_square(U, sub);
  const auto vv = post::plane_average_square(V, sub);
  const auto ww = post::plane_average_square(W, sub);
  const auto tt = post::plane_average_square(T, sub);
  const auto uw = post::plane_average_product(U, W, sub);
  const auto ut = post::plane_average_product(U, T, sub);
  const auto wt = post::plane_average_product(W, T, sub);
  for (int q = 0; q < c_.nz; ++q) {
    sum_u_[q] += double(u_bar[q]) * weight;
    sum_v_[q] += double(v_bar[q]) * weight;
    sum_w_[q] += double(w_bar[q]) * weight;
    sum_t_[q] += double(t_bar[q]) * weight;
    sum_uu_[q] += double(uu[q]) * weight;
    sum_vv_[q] += double(vv[q]) * weight;
    sum_ww_[q] += double(ww[q]) * weight;
    sum_tt_[q] += double(tt[q]) * weight;
    sum_uw_[q] += double(uw[q]) * weight;
    sum_ut_[q] += double(ut[q]) * weight;
    sum_wt_[q] += double(wt[q]) * weight;
  }
}

void Statistics::sample_conditional(const real_t weight, const Field& U, const Field& W,
                                    const Field& T, const mpmstd::grid::Grid& grid,
                                    const Subdomain& sub) {
  namespace post = mpmstd::post;
  for (int side = 0; side < 2; ++side) {
    const post::WallRichardsonResult r = post::classify_wall_richardson(
        U, T, grid, sub, c_.nu, c_.alpha, side, static_cast<real_t>(c_.richardson_threshold),
        static_cast<real_t>(c_.richardson_z_plus), bins_, weight_low_[side]);
    post::complement_weights(weight_high_[side], weight_low_[side]);
    rho_sum_[side] += double(r.area_fraction) * weight;
    for (std::size_t b = 0; b < r.histogram.size(); ++b) histogram_[side][b] += r.histogram[b] * weight;
    for (int cls = 0; cls < 2; ++cls) {
      const mpmstd::field::Buffer& w = cls == 0 ? weight_low_[side] : weight_high_[side];
      const std::vector<real_t> profiles[6] = {
          post::plane_weighted_average(U, w, sub),         post::plane_weighted_average(T, w, sub),
          post::plane_weighted_average_square(U, w, sub),  post::plane_weighted_average_square(T, w, sub),
          post::plane_weighted_average_product(U, W, w, sub),
          post::plane_weighted_average_product(W, T, w, sub)};
      for (int q = 0; q < 6; ++q)
        for (int k = 0; k < c_.nz; ++k) cond_sum_[side][cls][q][k] += double(profiles[q][k]) * weight;
    }
  }
}

void Statistics::sample_spectra(const real_t weight, const Field& U, const Subdomain& sub) {
  mpmstd::post::LineSpectra now;
  spectrum_plan_->compute(U, sub, now);
  if (spectra_sum_.x.empty()) {
    spectra_sum_ = now;
    for (real_t& v : spectra_sum_.x) v *= weight;
    for (real_t& v : spectra_sum_.y) v *= weight;
    return;
  }
  for (std::size_t m = 0; m < now.x.size(); ++m) spectra_sum_.x[m] += now.x[m] * weight;
  for (std::size_t m = 0; m < now.y.size(); ++m) spectra_sum_.y[m] += now.y[m] * weight;
}

int Statistics::write(const long step, const double t, const bool is_root) const {
  if (!is_root || sum_dt_ <= 0) return 0;
  int n = write_profiles(step, t);
  if (cond_sum_dt_ > 0) {
    n += write_conditional(step, t);
    n += write_richardson_pdf(step, t);
  }
  if (spec_sum_dt_ > 0) n += write_spectra(step, t);
  return n;
}

int Statistics::write_profiles(const long step, const double t) const {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/dhvc_stats_%08ld.dat", c_.stats_dir.c_str(), step);
  std::FILE* f = std::fopen(path, "w");
  if (!f) return 0;
  std::vector<double> u_mean, v_mean, w_mean, t_mean, uu_mean, vv_mean, ww_mean, tt_mean, uw_mean,
      ut_mean, wt_mean;
  mean_profile(sum_u_, u_mean);
  mean_profile(sum_v_, v_mean);
  mean_profile(sum_w_, w_mean);
  mean_profile(sum_t_, t_mean);
  mean_profile(sum_uu_, uu_mean);
  mean_profile(sum_vv_, vv_mean);
  mean_profile(sum_ww_, ww_mean);
  mean_profile(sum_tt_, tt_mean);
  mean_profile(sum_uw_, uw_mean);
  mean_profile(sum_ut_, ut_mean);
  mean_profile(sum_wt_, wt_mean);
  double u_max = 0, dstar = 0, nu_hot = 0, nu_cold = 0, ut_hot = 0, ut_cold = 0;
  int k_max = 0;
  maximum_and_displacement_thickness(u_mean, c_.z_center, u_max, dstar, k_max);
  wall_nusselt(t_mean, nu_hot, nu_cold);
  wall_friction_velocity(u_mean, ut_hot, ut_cold);
  const double re_dstar = u_max > 0 ? u_max * dstar / c_.nu : 0.0;
  // 논문 (7.1): Re = U_rms(δ_max)·H/ν, U_rms = √⟨u'u'⟩ at z = δ_max (= z(u_max)), H = lz
  const double u_rms_delta_max = rms_of(uu_mean[k_max], u_mean[k_max]);
  const double re_rms = u_rms_delta_max * c_.lz / c_.nu;
  // θ_τ = −f_w/u_τ (f_w = Nu·α·ΔT/lz), 평균량의 |Ri_f| = z_ref·ν·f_w/u_τ⁴ (gβ = 1)
  const double delta_t = c_.wall_temperature_hot - c_.wall_temperature_cold;
  const double fw_hot = nu_hot * c_.alpha * delta_t / c_.lz, fw_cold = nu_cold * c_.alpha * delta_t / c_.lz;
  const double theta_tau_hot = ut_hot > 0 ? fw_hot / ut_hot : 0.0;
  const double theta_tau_cold = ut_cold > 0 ? fw_cold / ut_cold : 0.0;
  const double ri_hot = ut_hot > 0 ? c_.richardson_z_plus * c_.nu * fw_hot / std::pow(ut_hot, 4) : 0.0;
  const double ri_cold = ut_cold > 0 ? c_.richardson_z_plus * c_.nu * fw_cold / std::pow(ut_cold, 4) : 0.0;
  std::fprintf(f, "# dhvc statistics  step=%ld t=%.4f t_start=%.4f sum_dt=%.4f sample_interval=%d\n", step,
               t, time_begin_, sum_dt_, c_.sample_interval);
  std::fprintf(f, "# Ra=%g Pr=%g nu=%.8e alpha=%.8e Re_dstar=%.6f u_max=%.6f delta_star=%.6e delta_max=%.6e\n",
               c_.rayleigh, c_.prandtl, c_.nu, c_.alpha, re_dstar, u_max, dstar, c_.z_center[k_max]);
  std::fprintf(f, "# Nu_hot=%.6f Nu_cold=%.6f u_tau_hot=%.8f u_tau_cold=%.8f theta_tau_hot=%.8e theta_tau_cold=%.8e\n",
               nu_hot, nu_cold, ut_hot, ut_cold, theta_tau_hot, theta_tau_cold);
  std::fprintf(f, "# U_rms_delta_max=%.6e Re_rms=%.6f Ri_f_hot=%.6e Ri_f_cold=%.6e mean_dPdx=%.8e velocity_ghost=%s\n",
               u_rms_delta_max, re_rms, ri_hot, ri_cold, sum_dpdx_ / sum_dt_,
               c_.mirror_wall ? "mirror" : "value");
  std::fprintf(f, "# columns: dt-weighted xy-plane averages; uw uT wT are slot products "
                  "(U x-face, W z-face, T center, no interpolation) minus mean products; rms = sqrt(<f^2>-<f>^2)\n");
  std::fprintf(f, "# z U V W T urms vrms wrms trms uw uT wT\n");
  for (int q = 0; q < c_.nz; ++q)
    std::fprintf(f, "%12.8f %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e\n",
                 c_.z_center[q], u_mean[q], v_mean[q], w_mean[q], t_mean[q],
                 rms_of(uu_mean[q], u_mean[q]), rms_of(vv_mean[q], v_mean[q]),
                 rms_of(ww_mean[q], w_mean[q]), rms_of(tt_mean[q], t_mean[q]),
                 uw_mean[q] - u_mean[q] * w_mean[q], ut_mean[q] - u_mean[q] * t_mean[q],
                 wt_mean[q] - w_mean[q] * t_mean[q]);
  std::fclose(f);
  std::printf("# stats written: %s  (Re_dstar=%.4f Nu=%.3f/%.3f)\n", path, re_dstar, nu_hot, nu_cold);
  return 1;
}

int Statistics::write_conditional(const long step, const double t) const {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/dhvc_conditional_%08ld.dat", c_.stats_dir.c_str(), step);
  std::FILE* f = std::fopen(path, "w");
  if (!f) return 0;
  // 조건 평균 프로파일 (가중 Σ/Σdt) — 논문 (6.2) 의 조건평균 그대로 (면적 정규화 포함)
  std::vector<double> mean[2][2][6];
  for (int side = 0; side < 2; ++side)
    for (int cls = 0; cls < 2; ++cls)
      for (int q = 0; q < 6; ++q) {
        mean[side][cls][q].resize(c_.nz);
        for (int k = 0; k < c_.nz; ++k) mean[side][cls][q][k] = cond_sum_[side][cls][q][k] / cond_sum_dt_;
      }
  // 국소 Nu_l/Nu_h (논문 (7.1): f_w,l = −κ dΘ̄_l/dz|_w) — 조건 T 프로파일의 벽 기울기
  const double delta_t = c_.wall_temperature_hot - c_.wall_temperature_cold;
  const auto nusselt_hot = [&](const std::vector<double>& tm) {
    return (c_.wall_temperature_hot - tm[0]) / z_first_ * c_.lz / delta_t;
  };
  const auto nusselt_cold = [&](const std::vector<double>& tm) {
    return (tm[c_.nz - 1] - c_.wall_temperature_cold) / z_last_gap_ * c_.lz / delta_t;
  };
  std::fprintf(f, "# dhvc conditional statistics  step=%ld t=%.4f t_start=%.4f sum_dt=%.4f\n", step, t,
               time_begin_, cond_sum_dt_);
  std::fprintf(f, "# richardson_threshold=%g z_plus_reference=%g rho_hot=%.6f rho_cold=%.6f\n",
               c_.richardson_threshold, c_.richardson_z_plus, rho_sum_[0] / cond_sum_dt_,
               rho_sum_[1] / cond_sum_dt_);
  // 비어 있는 class(면적비 0)의 조건평균은 0 이라 Nu 가 가짜값이 된다 → nan
  const double rho_hot = rho_sum_[0] / cond_sum_dt_, rho_cold = rho_sum_[1] / cond_sum_dt_;
  const double nan = std::nan("");
  std::fprintf(f, "# Nu_low_hot=%.6f Nu_high_hot=%.6f Nu_low_cold=%.6f Nu_high_cold=%.6f\n",
               rho_hot > 1e-12 ? nusselt_hot(mean[0][0][1]) : nan,
               1 - rho_hot > 1e-12 ? nusselt_hot(mean[0][1][1]) : nan,
               rho_cold > 1e-12 ? nusselt_cold(mean[1][0][1]) : nan,
               1 - rho_cold > 1e-12 ? nusselt_cold(mean[1][1][1]) : nan);
  std::fprintf(f, "# classification per wall column by local |Ri_f| = z_ref*nu*f_w/u_tau^4 (gbeta=1): "
                  "low = 0<|Ri_f|<=threshold (streaky high-shear), high = otherwise. Conditional averages "
                  "normalised by the conditional area (paper eq. 6.2); uu tt are raw <u^2>_c <T^2>_c "
                  "(paper eq. 6.4 applies rho), uw wT are slot products.\n");
  std::fprintf(f, "# z");
  for (const char* side : {"hot", "cold"})
    for (const char* cls : {"low", "high"})
      for (const char* q : {"U", "T", "uu", "tt", "uw", "wT"}) std::fprintf(f, " %s_%s_%s", q, cls, side);
  std::fprintf(f, "\n");
  for (int k = 0; k < c_.nz; ++k) {
    std::fprintf(f, "%12.8f", c_.z_center[k]);
    for (int side = 0; side < 2; ++side)
      for (int cls = 0; cls < 2; ++cls)
        for (int q = 0; q < 6; ++q) std::fprintf(f, " %14.8e", mean[side][cls][q][k]);
    std::fprintf(f, "\n");
  }
  std::fclose(f);
  return 1;
}

int Statistics::write_richardson_pdf(const long step, const double t) const {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/dhvc_richardson_pdf_%08ld.dat", c_.stats_dir.c_str(), step);
  std::FILE* f = std::fopen(path, "w");
  if (!f) return 0;
  const double width = (bins_.log10_max - bins_.log10_min) / bins_.count;
  std::fprintf(f, "# dhvc |Ri_f| histogram  step=%ld t=%.4f sum_dt=%.4f bins=%d log10_min=%g log10_max=%g\n",
               step, t, cond_sum_dt_, bins_.count, bins_.log10_min, bins_.log10_max);
  std::fprintf(f, "# underflow_hot=%.6e overflow_hot=%.6e underflow_cold=%.6e overflow_cold=%.6e "
                  "(dt-weighted column counts; overflow includes u_tau=0)\n",
               histogram_[0][0], histogram_[0][bins_.count + 1], histogram_[1][0],
               histogram_[1][bins_.count + 1]);
  std::fprintf(f, "# log10_Rif_center count_hot count_cold   (PDF over log10: count/(total*bin_width); paper fig.4)\n");
  for (int b = 0; b < bins_.count; ++b)
    std::fprintf(f, "%10.4f %14.6e %14.6e\n", bins_.log10_min + (b + 0.5) * width, histogram_[0][b + 1],
                 histogram_[1][b + 1]);
  std::fclose(f);
  return 1;
}

int Statistics::write_spectra(const long step, const double t) const {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/dhvc_spectra_%08ld.dat", c_.stats_dir.c_str(), step);
  std::FILE* f = std::fopen(path, "w");
  if (!f) return 0;
  const mpmstd::post::LineSpectra& s = spectra_sum_;
  std::fprintf(f, "# dhvc 1-D energy spectra of u (dt-weighted)  step=%ld t=%.4f sum_dt=%.4f nz=%d n_mode_x=%d "
                  "n_mode_y=%d lx=%g ly=%g\n",
               step, t, spec_sum_dt_, s.nz_global, s.n_mode_x, s.n_mode_y, c_.lx, c_.ly);
  std::fprintf(f, "# E(k; z): one-sided, sum_m E[m] = <u^2>_line (m=0 holds the mean^2); k_x = 2*pi*m/lx, "
                  "k_y = 2*pi*m/ly. Block 'x': rows z, columns m = 0..n_mode_x-1; block 'y' likewise.\n");
  std::fprintf(f, "# block x\n");
  for (int k = 0; k < s.nz_global; ++k) {
    std::fprintf(f, "%12.8f", c_.z_center[k]);
    for (int m = 0; m < s.n_mode_x; ++m)
      std::fprintf(f, " %.6e", s.x[static_cast<std::size_t>(k) * s.n_mode_x + m] / spec_sum_dt_);
    std::fprintf(f, "\n");
  }
  std::fprintf(f, "# block y\n");
  for (int k = 0; k < s.nz_global; ++k) {
    std::fprintf(f, "%12.8f", c_.z_center[k]);
    for (int m = 0; m < s.n_mode_y; ++m)
      std::fprintf(f, " %.6e", s.y[static_cast<std::size_t>(k) * s.n_mode_y + m] / spec_sum_dt_);
    std::fprintf(f, "\n");
  }
  std::fclose(f);
  return 1;
}

}  // namespace dhvc
