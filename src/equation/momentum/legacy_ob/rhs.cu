#include "equation/momentum/legacy_ob/rhs.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/transport_stencil.hpp"

// GPU 백엔드 — rhs.cpp 의 쌍둥이. 수식은 momentum_stencil.hpp 하나를 같이 쓰고,
// 여기는 커널 발사만 있다. GPU 빌드는 rhs.cpp 대신 이 파일을 담는다 (Makefile).

namespace mpmstd::equation::momentum {

namespace {

__global__ void add_kernel(const field::View rhs, const field::ConstView q,
                           const field::ConstView U, const field::ConstView V,
                           const field::ConstView W, const real_t* dx_x,
                           const real_t* dmx_x, const real_t* dx_y,
                           const real_t* dmx_y, const real_t* dx_z,
                           const real_t* dmx_z, const int h, const int ni,
                           const int nj, const int nk, const real_t nu_dt,
                           const real_t dt, const bool w_is_z_face) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  rhs(i, j, k) += equation::stencil::advection_diffusion(q, U, V, W, i, j, k, dx_x, dmx_x,
                                               dx_y, dmx_y, dx_z, dmx_z, nu_dt, dt,
                                               w_is_z_face);
}

void add_impl(const char* who, field::Field& rhs, const field::Field& q,
              const field::Field& U, const field::Field& V, const field::Field& W,
              const grid::Grid& grid, const parallel::Subdomain& sub,
              const real_t viscosity, const real_t dt, const bool w_is_z_face) {
  check_field_matches(who, rhs, sub);
  check_field_matches(who, q, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);

  const int h = halo_width;
  const int ni = sub.x().n_local;
  const int nj = sub.y().n_local;
  const int nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(
      rhs.view(), q.view(), U.view(), V.view(), W.view(),
      grid.x.device_cell_width.data(), grid.x.device_center_distance.data(),
      grid.y.device_cell_width.data(), grid.y.device_center_distance.data(),
      grid.z.device_cell_width.data(), grid.z.device_center_distance.data(), h, ni,
      nj, nk, viscosity * dt, dt, w_is_z_face);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void add_advection_diffusion_cell(field::Field& rhs, const field::Field& q,
                                  const field::Field& U, const field::Field& V,
                                  const field::Field& W, const grid::Grid& grid,
                                  const parallel::Subdomain& sub,
                                  const real_t viscosity, const real_t dt) {
  add_impl("add_advection_diffusion_cell", rhs, q, U, V, W, grid, sub, viscosity, dt,
           /*w_is_z_face=*/false);
}

void add_advection_diffusion_z_face(field::Field& rhs, const field::Field& w,
                                    const field::Field& U, const field::Field& V,
                                    const field::Field& W, const grid::Grid& grid,
                                    const parallel::Subdomain& sub,
                                    const real_t viscosity, const real_t dt) {
  add_impl("add_advection_diffusion_z_face", rhs, w, U, V, W, grid, sub, viscosity, dt,
           /*w_is_z_face=*/true);
}

}  // namespace mpmstd::equation::momentum
