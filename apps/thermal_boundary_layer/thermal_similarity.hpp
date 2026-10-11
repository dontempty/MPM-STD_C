#pragma once
// 층류 평판 열 경계층 상사해 (Blasius 기반) — 열 경계층 앱의 유입 프로파일·초기조건 (2026-10-11).
// 무차원: Li et al. (2009) eq. (5)(6) — 길이 δ*₀, 속도 U∞, 스칼라 θ ∈ [θ_w, 1].
//
//   isoscalar (θ_w 고정): θ″ + (Pr/2) f θ′ = 0, θ(0) = 0, θ(∞) = 1
//                        → θ(η) = ∫₀^η e^{−(Pr/2)∫₀^s f} ds / ∫₀^∞ (⋯)   (Pohlhausen)
//                        물리 θ = θ_w + (1 − θ_w)·θ(η)
//   isoflux (q_w 고정):   T − T∞ = (q_w/k) √(νx/U∞) g(η),  g″ + (Pr/2)(f g′ − f′ g) = 0,
//                        g′(0) = −1, g(∞) = 0 (선형 ODE → 두 기본해 결합)
//                        Li 의 θ = 1 − k(T − T∞)/(q_w δ*₀) = 1 − √(x/x₀) g(η)/1.7208   (x₀: δ* = 1)
//                        → 벽에서 ∂θ/∂z = (1/1.7208)·√(x₀/x)·√(U∞/(νx))·… = √(x₀/x)·(1/δ*₀) = 1 at x = x₀ ✓
// η = z √(U∞/(ν x)). 표를 만들어 선형 보간한다 (Blasius 와 같은 격자 η).
#include <cmath>
#include <vector>

#include "blasius.hpp"

namespace boundary_layer {

struct ThermalSimilarity {
  double deta = 1e-3, eta_max = 12.0;
  std::vector<double> theta;  // isoscalar: θ(η) ∈ [0,1];  isoflux: g(η) (g(0) > 0, → 0)
  bool isoflux = false;
  double wall_coefficient = 0;  // isoflux: g(0); isoscalar: θ′(0)

  static ThermalSimilarity solve_isoscalar(const Blasius& b, const double prandtl) {
    ThermalSimilarity t;
    t.deta = b.deta; t.eta_max = b.eta_max; t.isoflux = false;
    const int n = static_cast<int>(t.eta_max / t.deta) + 1;
    // θ′ ∝ exp(−(Pr/2) ∫ f): 사다리꼴 누적
    std::vector<double> integral_f(n, 0.0), dtheta(n, 0.0);
    for (int i = 1; i < n; ++i)
      integral_f[i] = integral_f[i - 1] + 0.5 * t.deta * (b.f[i - 1] + b.f[i]);
    for (int i = 0; i < n; ++i) dtheta[i] = std::exp(-0.5 * prandtl * integral_f[i]);
    t.theta.assign(n, 0.0);
    for (int i = 1; i < n; ++i) t.theta[i] = t.theta[i - 1] + 0.5 * t.deta * (dtheta[i - 1] + dtheta[i]);
    const double norm = t.theta.back();
    for (double& v : t.theta) v /= norm;
    t.wall_coefficient = 1.0 / norm;  // θ′(0)
    return t;
  }

  static ThermalSimilarity solve_isoflux(const Blasius& b, const double prandtl) {
    ThermalSimilarity t;
    t.deta = b.deta; t.eta_max = b.eta_max; t.isoflux = true;
    const int n = static_cast<int>(t.eta_max / t.deta) + 1;
    // 선형 ODE: g = a·G1 + G2, G1: (1, 0), G2: (0, −1);  g(∞) = 0 → a = −G2(∞)/G1(∞)
    std::vector<double> g1 = integrate(b, prandtl, 1.0, 0.0), g2 = integrate(b, prandtl, 0.0, -1.0);
    const double a = -g2.back() / g1.back();
    t.theta.assign(n, 0.0);
    for (int i = 0; i < n; ++i) t.theta[i] = a * g1[i] + g2[i];
    t.wall_coefficient = t.theta[0];
    return t;
  }

  double value(const double eta) const {
    if (eta <= 0) return theta.front();
    if (eta >= eta_max) return isoflux ? 0.0 : 1.0;
    const double q = eta / deta;
    const int i = static_cast<int>(q);
    const double w = q - i;
    return (1 - w) * theta[i] + w * theta[i + 1];
  }

 private:
  // g″ = (Pr/2)(f′ g − f g′), RK4 on the Blasius table grid
  static std::vector<double> integrate(const Blasius& b, const double prandtl, const double g0,
                                       const double dg0) {
    const int n = static_cast<int>(b.eta_max / b.deta) + 1;
    std::vector<double> g(n, 0.0);
    double y0 = g0, y1 = dg0;
    g[0] = y0;
    const double hh = b.deta;
    const auto f_at = [&](const double eta) { return b.f_value(eta); };
    const auto fp_at = [&](const double eta) { return b.f_prime(eta); };
    for (int i = 1; i < n; ++i) {
      const double eta = (i - 1) * hh;
      const auto rhs = [&](double e, double a0, double a1, double& d0, double& d1) {
        d0 = a1;
        d1 = 0.5 * prandtl * (fp_at(e) * a0 - f_at(e) * a1);
      };
      double k[4][2];
      rhs(eta, y0, y1, k[0][0], k[0][1]);
      rhs(eta + 0.5 * hh, y0 + 0.5 * hh * k[0][0], y1 + 0.5 * hh * k[0][1], k[1][0], k[1][1]);
      rhs(eta + 0.5 * hh, y0 + 0.5 * hh * k[1][0], y1 + 0.5 * hh * k[1][1], k[2][0], k[2][1]);
      rhs(eta + hh, y0 + hh * k[2][0], y1 + hh * k[2][1], k[3][0], k[3][1]);
      y0 += hh / 6 * (k[0][0] + 2 * k[1][0] + 2 * k[2][0] + k[3][0]);
      y1 += hh / 6 * (k[0][1] + 2 * k[1][1] + 2 * k[2][1] + k[3][1]);
      g[i] = y0;
    }
    return g;
  }
};

}  // namespace boundary_layer
