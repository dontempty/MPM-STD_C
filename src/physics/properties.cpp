#include "physics/properties.hpp"

#include "common/field_checks.hpp"

// CPU 백엔드 — 수식은 properties.hpp 의 evaluate_properties 한 곳 (GPU 쌍둥이
// properties.cu). ghost 포함 전 셀: 면 보간 커널들이 벽·halo 바로 안쪽에서
// 이웃 물성을 읽기 때문이다 (T ghost 가 최신이면 물성 ghost 도 자동으로 맞다).

namespace mpmstd::physics {

void update_properties(PropertyFields& properties, const field::Field& T,
                       const PropertyModel& model, const parallel::Subdomain& sub) {
  check_field_matches("update_properties", T, sub);
  check_field_matches("update_properties", properties.viscosity, sub);
  const real_t* temperature = T.data();
  real_t* viscosity = properties.viscosity.data();
  real_t* inverse_density = properties.inverse_density.data();
  real_t* conductivity = properties.conductivity.data();
  real_t* conductivity_derivative = properties.conductivity_derivative.data();
  real_t* inverse_heat_capacity = properties.inverse_heat_capacity.data();
  real_t* inverse_heat_capacity_derivative =
      properties.inverse_heat_capacity_derivative.data();
  const std::size_t n = T.size();
  for (std::size_t q = 0; q < n; ++q) {
    const PropertyValues v = evaluate_properties(model, temperature[q]);
    viscosity[q] = v.viscosity;
    inverse_density[q] = v.inverse_density;
    conductivity[q] = v.conductivity;
    conductivity_derivative[q] = v.conductivity_derivative;
    inverse_heat_capacity[q] = v.inverse_heat_capacity;
    inverse_heat_capacity_derivative[q] = v.inverse_heat_capacity_derivative;
  }
}

}  // namespace mpmstd::physics
