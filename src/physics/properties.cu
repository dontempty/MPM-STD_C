#include "physics/properties.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"

// GPU 백엔드 — properties.cpp 의 쌍둥이. 수식은 properties.hpp 한 곳.

namespace mpmstd::physics {

namespace {

__global__ void update_kernel(const real_t* temperature, real_t* viscosity,
                              real_t* inverse_density, real_t* conductivity,
                              real_t* conductivity_derivative,
                              real_t* inverse_heat_capacity,
                              real_t* inverse_heat_capacity_derivative,
                              const PropertyModel model, const std::size_t n) {
  const std::size_t q = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (q >= n) return;
  const PropertyValues v = evaluate_properties(model, temperature[q]);
  viscosity[q] = v.viscosity;
  inverse_density[q] = v.inverse_density;
  conductivity[q] = v.conductivity;
  conductivity_derivative[q] = v.conductivity_derivative;
  inverse_heat_capacity[q] = v.inverse_heat_capacity;
  inverse_heat_capacity_derivative[q] = v.inverse_heat_capacity_derivative;
}

}  // namespace

void update_properties(PropertyFields& properties, const field::Field& T,
                       const PropertyModel& model, const parallel::Subdomain& sub) {
  check_field_matches("update_properties", T, sub);
  check_field_matches("update_properties", properties.viscosity, sub);
  const std::size_t n = T.size();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  update_kernel<<<blocks, threads>>>(
      T.data(), properties.viscosity.data(), properties.inverse_density.data(),
      properties.conductivity.data(), properties.conductivity_derivative.data(),
      properties.inverse_heat_capacity.data(),
      properties.inverse_heat_capacity_derivative.data(), model, n);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::physics
