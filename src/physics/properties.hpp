#pragma once
// NOB 온도 의존 물성 (글리세롤 다항식) — PaScaL_TCS module_global.f90 1:1.
//
// 모든 물성은 인자 t = ΔT·θ (θ ∈ [−0.5, 0.5]) 의 다항식이고, 기준온도 θm 은
// 계수에 이미 구워져 있다 (선행 계수 *[0] 로 정규화). OB 환원 두 극한의 차이:
//   * 기본 생성 PropertyModel (선행 계수만 1) → **모든** 출력 ≡ 1, 미분 ≡ 0
//     — 정확한 OB (가변 커널 전체가 상수물성 커널과 같은 고정점을 공유)
//   * glycerol(0) → t = 0 이라 κ̂·μ̂·1/ρ̂ ≡ 1 이지만 **미분 필드는 상수 c₁ 등으로
//     남는다** (참조 dKPP 는 dκ̂/dt). 벽 행에선 R(마스크 전)·밴드(마스크 후)의
//     선형화 소거가 설계상 깨지므로 glycerol(0) 은 '거의' OB 일 뿐이다 —
//     참조 스킴의 성질 그대로 (test_rbc_nob 주석).
// 그래서 상수물성(OB) 앱은 이 모듈을 아예 쓰지 않는다 — PropertyFields 는
// NOB 앱만 할당한다 (DHVC 등 OB 검증 경로는 기존 상수물성 커널 그대로).
//
//   ρ̂(t)  = a₀(1 + a₁t + a₂t² + … + a₅t⁵) / a₀      (density)
//   Ĉp(t) = b-다항식,  κ̂(t) = c-다항식,  μ̂ = ρ-다항식 · d-다항식 / (a₀d₀)
//
// 참조: MPM-STD_C physics/properties (docs/NOB_RBC_SPEC.md), Fig-9 계수 =
// PaScaL_TCS module_global.f90 l.123-126.

#include "common/hd.hpp"
#include "common/types.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics {

struct PropertyModel {
  // 계수 이름은 참조 그대로 a/b/c/d — 수식 대조가 1:1 이 되도록 유지한다.
  real_t a[6] = {1, 0, 0, 0, 0, 0};  // 밀도 ρ
  real_t b[6] = {1, 0, 0, 0, 0, 0};  // 비열 Cp
  real_t c[6] = {1, 0, 0, 0, 0, 0};  // 열전도도 κ
  real_t d[6] = {1, 0, 0, 0, 0, 0};  // 점성 인자 (μ = ρ-다항식 · d-다항식)
  real_t delta_temperature = 0;      // 차원 ΔT — 다항식 인자의 스케일

  static PropertyModel glycerol(const real_t delta_temperature) {
    PropertyModel m;
    m.a[0] = 1.2477e3;  m.a[1] = -4.789e-4;   m.a[2] = -0.3795e-6;
    m.b[0] = 2.5108e3;  m.b[1] = 22.511e-4;
    m.c[0] = 2.9351e-3; m.c[1] = 3.863e-4;
    m.d[0] = 238.71e-6; m.d[1] = -702.83e-4;  m.d[2] = 2393.1e-6;
    m.d[3] = -6923.0e-8; m.d[4] = 33131.3e-10; m.d[5] = -71517.5e-12;
    m.delta_temperature = delta_temperature;
    return m;
  }

  // 부력 비선형 몫 a₂/a₁ (ρ(T) 2차항; physics/buoyancy add_buoyancy_z_variable)
  real_t density_quadratic_ratio() const {
    return (a[2] > -1e-14 && a[2] < 1e-14) ? real_t(0) : a[2] / a[1];
  }
};

// Horner: k₀·(1 + k₁t + k₂t² + k₃t³ + k₄t⁴ + k₅t⁵)
MPMSTD_HD inline double property_polynomial(const real_t k[6], const double t) {
  return double(k[0])
         * (1.0 + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * k[5])))));
}

// 한 점의 물성 — PaScaL_TCS module_solve_momentum.f90:165-168 /
// module_solve_thermal.f90:155-168 과 행 단위 일치. 정규화 기준은
// Rho0 = a₀, Cp0 = b₀, KPP0 = c₀, Mu0 = a₀·d₀.
struct PropertyValues {
  real_t viscosity;                         // μ̂            (참조 Mu)
  real_t inverse_density;                   // 1/ρ̂          (참조 invRho)
  real_t conductivity;                      // κ̂            (참조 KPP)
  real_t conductivity_derivative;           // dκ̂/dt        (참조 dKPP)
  real_t inverse_heat_capacity;             // 1/(ρ̂Ĉp)      (참조 invRhoCp)
  real_t inverse_heat_capacity_derivative;  // d[1/(ρ̂Ĉp)]/dt (참조 dinvRhoCp)
};

MPMSTD_HD inline PropertyValues evaluate_properties(const PropertyModel& m,
                                                    const real_t theta) {
  const real_t* a = m.a;
  const real_t* b = m.b;
  const real_t* c = m.c;
  const real_t* d = m.d;
  const double rho0 = double(a[0]);
  const double cp0 = double(b[0]);
  const double kpp0 = double(c[0]);
  const double mu0 = double(a[0]) * d[0];
  const double t = double(m.delta_temperature) * double(theta);

  PropertyValues v;
  v.viscosity = static_cast<real_t>(property_polynomial(a, t)
                                    * property_polynomial(d, t) / mu0);
  v.inverse_density = static_cast<real_t>(rho0 / property_polynomial(a, t));
  v.conductivity = static_cast<real_t>(property_polynomial(c, t) / kpp0);
  v.conductivity_derivative = static_cast<real_t>(
      double(c[0])
      * (c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * (4.0 * c[4] + t * 5.0 * c[5]))))
      / kpp0);
  v.inverse_heat_capacity = static_cast<real_t>(
      rho0 * cp0 / property_polynomial(a, t) / property_polynomial(b, t));

  // 해석 미분 — 참조 자체가 분모를 3차에서 절단한다 (thermal.f90:162-168 그대로).
  const double ia = 1.0 + t * (a[1] + t * (a[2] + t * a[3]));
  const double ib = 1.0 + t * (b[1] + t * (b[2] + t * b[3]));
  const double t2 = t * t, t3 = t2 * t;
  const double numerator =
      b[1] * (1.0 + 3.0 * a[2] * t2 + 4.0 * a[3] * t3)
      + a[1] * (1.0 + 2.0 * b[1] * t + 3.0 * b[2] * t2 + 4.0 * b[3] * t3)
      + t * (b[2] * (2.0 + 5.0 * a[3] * t3)
             + a[2] * (2.0 + 4.0 * b[2] * t2 + 5.0 * b[3] * t3)
             + 3.0 * t * (a[3] + b[3] + 2.0 * a[3] * b[3] * t3));
  v.inverse_heat_capacity_derivative =
      static_cast<real_t>(-numerator / (ia * ia) / (ib * ib));
  return v;
}

// T 로부터 만든 물성 필드 6장 — 전부 셀 중심, ghost 포함 전체 배열.
// (물성은 T 의 점별 함수라 T 의 ghost 가 최신이면 별도 halo 교환이 필요 없다.)
// NOB 앱 전용 — 상수물성 앱은 할당 자체를 하지 않는다 (2026-10-04 사용자 지시).
struct PropertyFields {
  field::Field viscosity, inverse_density, conductivity, conductivity_derivative,
      inverse_heat_capacity, inverse_heat_capacity_derivative;
  PropertyFields(const int nx_total, const int ny_total, const int nz_total)
      : viscosity(nx_total, ny_total, nz_total, "viscosity"),
        inverse_density(nx_total, ny_total, nz_total, "inverse_density"),
        conductivity(nx_total, ny_total, nz_total, "conductivity"),
        conductivity_derivative(nx_total, ny_total, nz_total,
                                "conductivity_derivative"),
        inverse_heat_capacity(nx_total, ny_total, nz_total, "inverse_heat_capacity"),
        inverse_heat_capacity_derivative(nx_total, ny_total, nz_total,
                                         "inverse_heat_capacity_derivative") {}
};

// 매 스텝 Tⁿ 에서 물성 갱신 (ghost 포함 전 셀) — CPU 는 .cpp, GPU 는 .cu 쌍둥이.
void update_properties(PropertyFields& properties, const field::Field& T,
                       const PropertyModel& model, const parallel::Subdomain& sub);

}  // namespace mpmstd::physics
