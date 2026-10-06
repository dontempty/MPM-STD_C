#pragma once
// Blasius 상사해 — 2f‴ + f f″ = 0, f(0) = f′(0) = 0, f′(∞) = 1 (f″(0) = 0.332057).
//   η = z √(U∞/(ν x)),  u = U∞ f′(η),  w = ½ √(ν U∞/x) (η f′ − f)
//   δ* = 1.7208 √(νx/U∞),  θ = 0.66412 √(νx/U∞),  C_f = 0.664/√Re_x
// RK4 + 사격법(secant). 표를 만들어 선형 보간한다 (η_max 밖은 f′ = 1, f = f(η_max) + Δη).
#include <cmath>
#include <vector>

namespace boundary_layer {

struct Blasius {
  double deta = 1e-3, eta_max = 12.0;
  std::vector<double> f, fp;  // 격자 η_i = i·deta
  double shear = 0;           // f″(0)

  static Blasius solve(double eta_max = 12.0, double deta = 1e-3) {
    Blasius b;
    b.deta = deta;
    b.eta_max = eta_max;
    double s0 = 0.33, s1 = 0.34;
    double g0 = b.integrate(s0) - 1.0, g1 = b.integrate(s1) - 1.0;
    for (int it = 0; it < 50 && std::abs(g1) > 1e-13; ++it) {
      const double s2 = s1 - g1 * (s1 - s0) / (g1 - g0);
      s0 = s1; g0 = g1; s1 = s2; g1 = b.integrate(s1) - 1.0;
    }
    b.shear = s1;
    b.integrate(s1);  // 표 확정
    return b;
  }

  double f_prime(double eta) const { return table(fp, eta, 1.0, 0.0); }
  double f_value(double eta) const {
    if (eta >= eta_max) return f.back() + (eta - eta_max);
    return table(f, eta, 0.0, 0.0);
  }

 private:
  // 반환: f′(η_max); 표 f, fp 를 채운다
  double integrate(double s) {
    const int n = static_cast<int>(eta_max / deta) + 1;
    f.assign(n, 0.0);
    fp.assign(n, 0.0);
    double y0 = 0, y1 = 0, y2 = s;  // f, f′, f″
    f[0] = 0; fp[0] = 0;
    const auto rhs = [](double a0, double a1, double a2, double& d0, double& d1, double& d2) {
      d0 = a1; d1 = a2; d2 = -0.5 * a0 * a2;
    };
    for (int i = 1; i < n; ++i) {
      double k[4][3];
      rhs(y0, y1, y2, k[0][0], k[0][1], k[0][2]);
      rhs(y0 + 0.5 * deta * k[0][0], y1 + 0.5 * deta * k[0][1], y2 + 0.5 * deta * k[0][2], k[1][0], k[1][1], k[1][2]);
      rhs(y0 + 0.5 * deta * k[1][0], y1 + 0.5 * deta * k[1][1], y2 + 0.5 * deta * k[1][2], k[2][0], k[2][1], k[2][2]);
      rhs(y0 + deta * k[2][0], y1 + deta * k[2][1], y2 + deta * k[2][2], k[3][0], k[3][1], k[3][2]);
      y0 += deta / 6 * (k[0][0] + 2 * k[1][0] + 2 * k[2][0] + k[3][0]);
      y1 += deta / 6 * (k[0][1] + 2 * k[1][1] + 2 * k[2][1] + k[3][1]);
      y2 += deta / 6 * (k[0][2] + 2 * k[1][2] + 2 * k[2][2] + k[3][2]);
      f[i] = y0; fp[i] = y1;
    }
    return y1;
  }
  double table(const std::vector<double>& v, double eta, double beyond, double) const {
    if (eta <= 0) return v.front();
    if (eta >= eta_max) return beyond == 0.0 ? v.back() : beyond;
    const double q = eta / deta;
    const int i = static_cast<int>(q);
    const double w = q - i;
    return (1 - w) * v[i] + w * v[i + 1];
  }
};

}  // namespace boundary_layer
