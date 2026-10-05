#include "equation/momentum/rhs/rhs_variable.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/momentum/momentum_variable_stencil.hpp"

// GPU 백엔드 — rhs_variable.cpp 의 쌍둥이. 수식은 momentum_variable_stencil.hpp.

namespace mpmstd::equation::momentum {

namespace {

// component: 0=U, 1=V, 2=W (성분별 셀 함수 선택만 — 수식은 스텐실 한 곳)
template <class MuArray, class RhoArray>
__global__ void add_kernel(real_t* rhs, const real_t* U, const real_t* V,
                           const real_t* W, const MuArray viscosity,
                           const RhoArray inverse_density, const real_t* dx1,
                           const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
                           const real_t* dx3, const real_t* dmx3, const int h,
                           const int ni, const int nj, const int nk,
                           const int ny_total, const int nz_total, const real_t Cmu,
                           const real_t dt, const int component) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  real_t value;
  if (component == 0)
    value = stencil::rhs_u_variable_cell(i, j, k, ny_total, nz_total, U, V, W,
                                         viscosity, inverse_density, dx1, dmx1, dx2,
                                         dmx2, dx3, dmx3, Cmu, dt);
  else if (component == 1)
    value = stencil::rhs_v_variable_cell(i, j, k, ny_total, nz_total, U, V, W,
                                         viscosity, inverse_density, dx1, dmx1, dx2,
                                         dmx2, dx3, dmx3, Cmu, dt);
  else
    value = stencil::rhs_w_variable_cell(i, j, k, ny_total, nz_total, U, V, W,
                                         viscosity, inverse_density, dx1, dmx1, dx2,
                                         dmx2, dx3, dmx3, Cmu, dt);
  rhs[stencil::vindex(i, j, k, ny_total, nz_total)] += value;
}

void add_impl(const char* who, field::Field& rhs, const field::Field& U,
              const field::Field& V, const field::Field& W,
              const field::Field& viscosity, const field::Field& inverse_density,
              const grid::Grid& grid, const parallel::Subdomain& sub, const real_t Cmu,
              const real_t dt, const int component) {
  check_field_matches(who, rhs, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);
  check_field_matches(who, viscosity, sub);
  check_field_matches(who, inverse_density, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(
      rhs.data(), U.data(), V.data(), W.data(), viscosity.data(),
      inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, ni, nj, nk,
      sub.y().n_total, sub.z().n_total, Cmu, dt, component);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_impl_constant(const char* who, field::Field& rhs, const field::Field& U,
                       const field::Field& V, const field::Field& W,
                       const grid::Grid& grid, const parallel::Subdomain& sub,
                       const real_t viscosity, const real_t dt, const int component) {
  check_field_matches(who, rhs, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(
      rhs.data(), U.data(), V.data(), W.data(), equation::UnitArray{},
      equation::UnitArray{}, grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, ni, nj, nk,
      sub.y().n_total, sub.z().n_total, viscosity, dt, component);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void add_advection_diffusion_u_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_impl("add_advection_diffusion_u_variable", rhs, U, V, W, viscosity,
           inverse_density, grid, sub, Cmu, dt, 0);
}

void add_advection_diffusion_v_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_impl("add_advection_diffusion_v_variable", rhs, U, V, W, viscosity,
           inverse_density, grid, sub, Cmu, dt, 1);
}

void add_advection_diffusion_w_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_impl("add_advection_diffusion_w_variable", rhs, U, V, W, viscosity,
           inverse_density, grid, sub, Cmu, dt, 2);
}

void add_advection_diffusion_u_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_impl_constant("add_advection_diffusion_u_constant_property", rhs, U, V, W,
                    grid, sub, viscosity, dt, 0);
}

void add_advection_diffusion_v_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_impl_constant("add_advection_diffusion_v_constant_property", rhs, U, V, W,
                    grid, sub, viscosity, dt, 1);
}

void add_advection_diffusion_w_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_impl_constant("add_advection_diffusion_w_constant_property", rhs, U, V, W,
                    grid, sub, viscosity, dt, 2);
}

}  // namespace mpmstd::equation::momentum
