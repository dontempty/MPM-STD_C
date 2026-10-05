#include "physics/buoyancy.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "physics/physics_stencil.hpp"

// GPU 백엔드 — buoyancy.cpp 의 쌍둥이.

namespace mpmstd::physics {

namespace {

enum : int { kThreads = 256 };

__global__ void buoyancy_kernel(const field::View rhs_w, const field::ConstView T,
                                const real_t* dz, const real_t* dmz, const int h,
                                const int i1, const int i2, const int i3,
                                const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  rhs_w(i, j, k) += dt * stencil::temperature_at_z_face(T, i, j, k, dz, dmz);
}

__global__ void buoyancy_variable_kernel(
    const field::View rhs_w, const field::ConstView T, const field::ConstView ir,
    const real_t* dz, const real_t* dmz, const int h, const int i1, const int i2,
    const int i3, const real_t density_quadratic_ratio,
    const real_t delta_temperature, const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const real_t T_face = stencil::temperature_at_z_face(T, i, j, k, dz, dmz);
  const real_t ir_face = stencil::temperature_at_z_face(ir, i, j, k, dz, dmz);
  rhs_w(i, j, k) += dt * ir_face
                    * (T_face
                       + density_quadratic_ratio * T_face * T_face
                             * delta_temperature);
}

}  // namespace

void add_buoyancy_z(field::Field& rhs_w, const field::Field& T, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("add_buoyancy_z", rhs_w, sub);
  check_field_matches("add_buoyancy_z", T, sub);
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  buoyancy_kernel<<<static_cast<int>((n + kThreads - 1) / kThreads), kThreads>>>(
      rhs_w.view(), T.view(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, sub.x().n_local,
      sub.y().n_local, sub.z().n_local, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_buoyancy_z_variable_property(field::Field& rhs_w, const field::Field& T,
                             const field::Field& inverse_density,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t density_quadratic_ratio,
                             const real_t delta_temperature, const real_t dt) {
  check_field_matches("add_buoyancy_z_variable_property", rhs_w, sub);
  check_field_matches("add_buoyancy_z_variable_property", T, sub);
  check_field_matches("add_buoyancy_z_variable_property", inverse_density, sub);
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  buoyancy_variable_kernel<<<static_cast<int>((n + kThreads - 1) / kThreads),
                             kThreads>>>(
      rhs_w.view(), T.view(), inverse_density.view(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, sub.x().n_local,
      sub.y().n_local, sub.z().n_local, density_quadratic_ratio, delta_temperature,
      dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::physics
