#include "rbc_nob/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "equation/sweep_kernels.hpp"
#include "post/statistics.hpp"

namespace rbc_nob {

using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;
namespace physics = mpmstd::physics;

namespace {
double rms_of(const double square_mean, const double mean) {
  return std::sqrt(std::max(0.0, square_mean - mean * mean));
}
// 다항식 정규화 값: k₀(1 + …)/k₀
double normalized_polynomial(const mpmstd::real_t k[6], const double t) {
  return physics::property_polynomial(k, t) / double(k[0]);
}
}  // namespace

Statistics::Statistics(const StatisticsConfig& config, const Subdomain& sub)
    : c_(config),
      mean_t_sum_(sub.x().n_total, sub.y().n_total, sub.z().n_total, "stats_mean_T_sum"),
      mean_w_sum_(sub.x().n_total, sub.y().n_total, sub.z().n_total, "stats_mean_W_sum"),
      sub_(&sub) {
  mean_t_sum_.fill(0);
  mean_w_sum_.fill(0);
  if (c_.nz <= 0 || static_cast<int>(c_.z_center.size()) != c_.nz
      || static_cast<int>(c_.dz.size()) != c_.nz)
    throw std::invalid_argument("rbc_nob::Statistics: z_center/dz size must equal nz");
  if (c_.sample_interval <= 0) throw std::invalid_argument("rbc_nob::Statistics: sample_interval > 0");
  // κ̂ at the wall temperatures (ΔT·θ = ±ΔT/2)
  kappa_wall_bottom_ = static_cast<real_t>(
      normalized_polynomial(c_.model.c, c_.delta_temperature * c_.wall_temperature_bottom));
  kappa_wall_top_ = static_cast<real_t>(
      normalized_polynomial(c_.model.c, c_.delta_temperature * c_.wall_temperature_top));
  z_first_ = c_.z_center[0];
  z_last_gap_ = static_cast<real_t>(c_.lz) - c_.z_center[c_.nz - 1];
  kc_low_ = c_.nz / 2 - 1;
  kc_high_ = c_.nz / 2;
  for (std::vector<double>* s : {&sum_t_, &sum_u_, &sum_v_, &sum_w_, &sum_uu_, &sum_vv_, &sum_ww_,
                                 &sum_tt_, &sum_wt_, &sum_mu_, &sum_kappa_})
    s->assign(c_.nz, 0.0);
  histogram_.assign(c_.pdf_bins + 2, 0.0);
  volume_ = c_.lx * c_.ly * c_.lz;
}

void Statistics::wall_nusselt(const std::vector<real_t>& t_bar, double& nu_bottom,
                              double& nu_top) const {
  // 무차원 ΔT = 1, lz 로 정규화. 벽 기울기 = (T₁ − T_wall)/z₁ (T ghost 는 거울이라 벽면이 정확)
  const double gradient_bottom = double(t_bar[0] - c_.wall_temperature_bottom) / z_first_;
  const double gradient_top = double(c_.wall_temperature_top - t_bar[c_.nz - 1]) / z_last_gap_;
  nu_bottom = -kappa_wall_bottom_ * gradient_bottom * c_.lz;
  nu_top = -kappa_wall_top_ * gradient_top * c_.lz;
}

double Statistics::center_temperature(const std::vector<real_t>& t_bar) const {
  return 0.5 * (double(t_bar[kc_low_]) + double(t_bar[kc_high_]));
}

void Statistics::mean(const std::vector<double>& sum, std::vector<double>& out) const {
  out.resize(c_.nz);
  for (int q = 0; q < c_.nz; ++q) out[q] = sum_dt_ > 0 ? sum[q] / sum_dt_ : 0.0;
}

double Statistics::nusselt_volume() const {
  return sum_dt_ > 0 ? 1.0 + (sum_wt_integral_ / sum_dt_) / c_.Ct : 0.0;
}

double Statistics::theta_c() const { return sum_dt_ > 0 ? sum_theta_c_ / sum_dt_ : 0.0; }

void Statistics::accumulate(const long step, const double t, const real_t dt, const Field& U,
                            const Field& V, const Field& W, const Field& T,
                            const physics::PropertyFields& properties,
                            const mpmstd::grid::Grid& grid, const Subdomain& sub) {
  if (time_begin_ < 0) time_begin_ = t - dt;
  grid_ = &grid;
  pending_dt_ += dt;
  if (step % c_.sample_interval != 0) return;
  namespace post = mpmstd::post;
  const double weight = pending_dt_;
  const auto t_bar = post::plane_average(T, sub);
  const auto u_bar = post::plane_average(U, sub);
  const auto v_bar = post::plane_average(V, sub);
  const auto w_bar = post::plane_average(W, sub);
  const auto uu = post::plane_average_square(U, sub);
  const auto vv = post::plane_average_square(V, sub);
  const auto ww = post::plane_average_square(W, sub);
  const auto tt = post::plane_average_square(T, sub);
  const auto wt = post::plane_average_product(W, T, sub);
  const auto mu = post::plane_average(properties.viscosity, sub);
  const auto kappa = post::plane_average(properties.conductivity, sub);
  double wt_integral = 0;
  for (int q = 0; q < c_.nz; ++q) {
    sum_t_[q] += double(t_bar[q]) * weight;
    sum_u_[q] += double(u_bar[q]) * weight;
    sum_v_[q] += double(v_bar[q]) * weight;
    sum_w_[q] += double(w_bar[q]) * weight;
    sum_uu_[q] += double(uu[q]) * weight;
    sum_vv_[q] += double(vv[q]) * weight;
    sum_ww_[q] += double(ww[q]) * weight;
    sum_tt_[q] += double(tt[q]) * weight;
    sum_wt_[q] += double(wt[q]) * weight;
    sum_mu_[q] += double(mu[q]) * weight;
    sum_kappa_[q] += double(kappa[q]) * weight;
    wt_integral += double(wt[q]) * c_.dz[q];
  }
  sum_wt_integral_ += wt_integral / c_.lz * weight;
  sum_theta_c_ += center_temperature(t_bar) * weight;
  const std::vector<double> bins = post::volume_histogram(
      T, grid, sub, c_.wall_temperature_top, c_.wall_temperature_bottom, c_.pdf_bins);
  for (std::size_t b = 0; b < bins.size(); ++b) histogram_[b] += bins[b] * weight;
  // 3D 시간평균 장 (표본 합; 내부 셀만)
  mpmstd::equation::add_increment(mean_t_sum_, T.data(), sub);
  mpmstd::equation::add_increment(mean_w_sum_, W.data(), sub);
  ++n_field_samples_;
  sum_dt_ += weight;
  pending_dt_ = 0;
}

int Statistics::write(const long step, const double t, const bool is_root) const {
  if (sum_dt_ <= 0) return 0;
  // ── 집합통신이 필요한 양은 루트 조기반환 **전에** 전 랭크가 계산한다 (교착 방지) ──
  double u_plume = 0;                 // U_pl = max |⟨w⟩_t| (4.25)
  std::vector<double> mean_histogram; // 시간평균 T 장의 체적 히스토그램 (합 장에 n 배 구간 → 스케일 커널 불필요)
  if (n_field_samples_ > 0 && grid_ && sub_) {
    const auto w_max = mpmstd::post::plane_maximum(mean_w_sum_, *sub_, /*absolute=*/true);
    for (const real_t v : w_max) u_plume = std::max(u_plume, double(v));
    u_plume /= double(n_field_samples_);
    const real_t n = static_cast<real_t>(n_field_samples_);
    mean_histogram = mpmstd::post::volume_histogram(mean_t_sum_, *grid_, *sub_, c_.wall_temperature_top * n,
                                                    c_.wall_temperature_bottom * n, c_.pdf_bins);
  }
  if (!is_root) return 0;
  std::vector<double> t_mean, u_mean, v_mean, w_mean, uu, vv, ww, tt, wt, mu, kappa;
  mean(sum_t_, t_mean); mean(sum_u_, u_mean); mean(sum_v_, v_mean); mean(sum_w_, w_mean);
  mean(sum_uu_, uu); mean(sum_vv_, vv); mean(sum_ww_, ww); mean(sum_tt_, tt); mean(sum_wt_, wt);
  mean(sum_mu_, mu); mean(sum_kappa_, kappa);
  const int nz = c_.nz;
  // rms·수평 rms·운동에너지
  std::vector<double> u_rms(nz), v_rms(nz), w_rms(nz), t_rms(nz), uh_rms(nz), ke(nz), nu_z(nz);
  for (int q = 0; q < nz; ++q) {
    u_rms[q] = rms_of(uu[q], u_mean[q]);
    v_rms[q] = rms_of(vv[q], v_mean[q]);
    w_rms[q] = rms_of(ww[q], w_mean[q]);
    t_rms[q] = rms_of(tt[q], t_mean[q]);
    uh_rms[q] = std::sqrt(u_rms[q] * u_rms[q] + v_rms[q] * v_rms[q]);
    ke[q] = 0.5 * (uu[q] + vv[q] + ww[q]);
  }
  // Nu(z) = [⟨wT⟩ − Ct·⟨κ̂⟩·dT̄/dz]/Ct   (전도항은 평균 κ̂ × 평균 기울기 근사)
  for (int q = 0; q < nz; ++q) {
    const int qm = std::max(0, q - 1), qp = std::min(nz - 1, q + 1);
    const double dtdz = (t_mean[qp] - t_mean[qm]) / (c_.z_center[qp] - c_.z_center[qm]);
    nu_z[q] = (wt[q] - c_.Ct * kappa[q] * dtdz) / c_.Ct;
  }
  // 벽 Nu·θ_c·열 BL(slope thickness)
  const double theta_c_mean = theta_c();
  std::vector<real_t> t_mean_real(t_mean.begin(), t_mean.end());
  double nu_bottom = 0, nu_top = 0;
  wall_nusselt(t_mean_real, nu_bottom, nu_top);
  const double gradient_bottom = std::abs(double(t_mean[0] - c_.wall_temperature_bottom) / z_first_);
  const double gradient_top = std::abs(double(c_.wall_temperature_top - t_mean[nz - 1]) / z_last_gap_);
  const double lambda_theta_b = gradient_bottom > 0 ? (c_.wall_temperature_bottom - theta_c_mean) / gradient_bottom : 0;
  const double lambda_theta_t = gradient_top > 0 ? (theta_c_mean - c_.wall_temperature_top) / gradient_top : 0;
  // 점성 BL: 수평 rms 의 첫 극대 (아래/위 벽에서 각각)
  int kb = 0;
  while (kb + 1 < nz && uh_rms[kb + 1] >= uh_rms[kb]) ++kb;
  int kt = nz - 1;
  while (kt - 1 >= 0 && uh_rms[kt - 1] >= uh_rms[kt]) --kt;
  const double lambda_u_b = c_.z_center[kb], lambda_u_t = c_.lz - c_.z_center[kt];
  // Re_E (체적 운동에너지), ν_m / ν_c 두 기준
  double ke_volume = 0;
  for (int q = 0; q < nz; ++q) ke_volume += ke[q] * c_.dz[q];
  ke_volume /= c_.lz;
  const double u_energy = std::sqrt(std::max(0.0, ke_volume));
  const double re_m = u_energy * c_.lz / c_.Cmu;
  const double t_c_dim = c_.delta_temperature * theta_c_mean;
  const double nu_hat_c = normalized_polynomial(c_.model.d, t_c_dim);   // ν̂ = μ̂/ρ̂ = d-다항식
  const double rho_hat_c = normalized_polynomial(c_.model.a, t_c_dim);
  const double re_c = nu_hat_c > 0 ? re_m / nu_hat_c : 0;
  const double re_pl_m = u_plume * c_.lz / c_.Cmu;            // Re_pl (4.24)(4.25), ν_m
  const double re_pl_c = nu_hat_c > 0 ? re_pl_m / nu_hat_c : 0;  // ν_c

  char path[512];
  std::snprintf(path, sizeof(path), "%s/rbc_nob_stats_%08ld.dat", c_.stats_dir.c_str(), step);
  std::FILE* f = std::fopen(path, "w");
  if (!f) return 0;
  std::fprintf(f, "# rbc_nob statistics  step=%ld t=%.4f t_start=%.4f sum_dt=%.4f sample_interval=%d\n",
               step, t, time_begin_, sum_dt_, c_.sample_interval);
  std::fprintf(f, "# Ra=%g Pr=%g DeltaT=%g Nu_volume=%.4f theta_c=%.5f Nu_bottom=%.4f Nu_top=%.4f Nu_wall=%.4f\n",
               c_.rayleigh, c_.prandtl, c_.delta_temperature, nusselt_volume(), theta_c_mean, nu_bottom,
               nu_top, 0.5 * (nu_bottom + nu_top));
  std::fprintf(f, "# lambda_theta_b=%.6f lambda_theta_t=%.6f chi_lambda_theta=%.4f lambda_u_b=%.6f lambda_u_t=%.6f chi_lambda_u=%.4f\n",
               lambda_theta_b, lambda_theta_t, lambda_theta_b > 0 ? lambda_theta_t / lambda_theta_b : 0,
               lambda_u_b, lambda_u_t, lambda_u_b > 0 ? lambda_u_t / lambda_u_b : 0);
  std::fprintf(f, "# U_E=%.6e Re_E_m=%.4f Re_E_c=%.4f nu_hat_c=%.4f rho_hat_c=%.4f mu_hat_c=%.4f kappa_wall_b=%.5f kappa_wall_t=%.5f T_rms_center=%.5f\n",
               u_energy, re_m, re_c, nu_hat_c, rho_hat_c, nu_hat_c * rho_hat_c, kappa_wall_bottom_,
               kappa_wall_top_, 0.5 * (t_rms[kc_low_] + t_rms[kc_high_]));
  std::fprintf(f, "# U_pl=%.6e Re_pl_m=%.4f Re_pl_c=%.4f n_field_samples=%ld   (U_pl = max|<w>_t| of the 3D time-mean W, "
                  "Horn (4.25); field means are sample-count weighted)\n",
               u_plume, re_pl_m, re_pl_c, n_field_samples_);
  std::fprintf(f, "# columns: dt-weighted xy-plane averages (Horn 2013 definitions); w_rms/wT_mean/ke on the MAC face below cell z (first row = wall face); Nu_z = (<wT> - Ct <kappa> dT/dz)/Ct "
                  "(conduction = mean kappa x mean gradient); lambda_theta = slope thickness (4.2)(4.3); "
                  "lambda_u = first maximum of horizontal rms (4.21)(4.22); Re_E from volume kinetic energy (4.26)\n");
  std::fprintf(f, "# z T_mean u_rms v_rms w_rms T_rms wT_mean Nu_z kappa_mean mu_mean ke_mean\n");
  for (int q = 0; q < nz; ++q)
    std::fprintf(f, "%12.8f %12.8f %12.6e %12.6e %12.6e %12.6e %14.8e %12.6f %12.8f %12.8f %12.6e\n",
                 c_.z_center[q], t_mean[q], u_rms[q], v_rms[q], w_rms[q], t_rms[q], wt[q], nu_z[q],
                 kappa[q], mu[q], ke[q]);
  std::fclose(f);
  std::printf("# stats written: %s  (theta_c=%.5f, Nu_vol=%.4f, Nu_wall b/t=%.4f/%.4f, chi_theta=%.3f, chi_u=%.3f)\n",
              path, theta_c_mean, nusselt_volume(), nu_bottom, nu_top,
              lambda_theta_b > 0 ? lambda_theta_t / lambda_theta_b : 0,
              lambda_u_b > 0 ? lambda_u_t / lambda_u_b : 0);
  int written = 1;

  std::snprintf(path, sizeof(path), "%s/rbc_nob_tpdf_%08ld.dat", c_.stats_dir.c_str(), step);
  f = std::fopen(path, "w");
  if (f) {
    const double width = (c_.wall_temperature_bottom - c_.wall_temperature_top) / c_.pdf_bins;
    double total = 0;
    for (const double v : histogram_) total += v;
    std::fprintf(f, "# rbc_nob temperature pdf  step=%ld t=%.4f sum_dt=%.4f bins=%d lower=%g upper=%g "
                    "(volume- and dt-weighted; pdf = fraction/width; Horn 2013 fig.8)\n",
                 step, t, sum_dt_, c_.pdf_bins, c_.wall_temperature_top, c_.wall_temperature_bottom);
    std::fprintf(f, "# underflow=%.6e overflow=%.6e (volume fraction)\n",
                 total > 0 ? histogram_[0] / total : 0, total > 0 ? histogram_[c_.pdf_bins + 1] / total : 0);
    double total_mean = 0;
    for (const double v : mean_histogram) total_mean += v;
    std::fprintf(f, "# mean-field pdf: histogram of the 3D time-mean T (n_field_samples=%ld; Horn fig.8 'time-averaged temperature'); "
                    "underflow=%.6e overflow=%.6e\n", n_field_samples_,
                 (total_mean > 0 && !mean_histogram.empty()) ? mean_histogram[0] / total_mean : 0,
                 (total_mean > 0 && !mean_histogram.empty()) ? mean_histogram[c_.pdf_bins + 1] / total_mean : 0);
    std::fprintf(f, "# T_center pdf_instant pdf_timemean\n");
    for (int b = 0; b < c_.pdf_bins; ++b)
      std::fprintf(f, "%10.5f %14.6e %14.6e\n", c_.wall_temperature_top + (b + 0.5) * width,
                   total > 0 ? histogram_[b + 1] / (total * width) : 0,
                   (total_mean > 0 && !mean_histogram.empty()) ? mean_histogram[b + 1] / (total_mean * width) : 0);
    std::fclose(f);
    ++written;
  }
  return written;
}

}  // namespace rbc_nob
