#include "equation/momentum/solve_kernels_variable.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/momentum/momentum_variable_stencil.hpp"

// GPU 백엔드 — solve_kernels_variable.cpp 의 쌍둥이.

namespace mpmstd::equation::momentum {

namespace {

struct Shape {
  int h, ni, nj, nk, ny_total, nz_total, blocks, threads;
};

Shape make_shape(const parallel::Subdomain& sub) {
  Shape s;
  s.h = halo_width;
  s.ni = sub.x().n_local;
  s.nj = sub.y().n_local;
  s.nk = sub.z().n_local;
  s.ny_total = sub.y().n_total;
  s.nz_total = sub.z().n_total;
  const long long n = static_cast<long long>(s.ni) * s.nj * s.nk;
  s.threads = 256;
  s.blocks = static_cast<int>((n + s.threads - 1) / s.threads);
  return s;
}

#define MPMSTD_CELL_FROM_IDX(shape)                                                   \
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; \
  if (idx >= static_cast<long long>(shape.ni) * shape.nj * shape.nk) return;          \
  const int k = shape.h + static_cast<int>(idx % shape.nk);                           \
  const int j = shape.h + static_cast<int>((idx / shape.nk) % shape.nj);              \
  const int i = shape.h                                                               \
                + static_cast<int>(idx / (static_cast<long long>(shape.nk) * shape.nj))

template <class MuArray, class RhoArray>
__global__ void couple_rhs_v_kernel(const Shape shape, real_t* rhs_v,
                                    const real_t* delta_u, const real_t* V,
                                    const MuArray mu, const RhoArray invrho,
                                    const real_t* dx1, const real_t* dmx1,
                                    const real_t* dx2, const real_t* dmx2,
                                    const real_t Cmu, const real_t dt) {
  MPMSTD_CELL_FROM_IDX(shape);
  rhs_v[stencil::vindex(i, j, k, shape.ny_total, shape.nz_total)] -=
      stencil::couple_rhs_v_from_delta_u_cell(i, j, k, shape.ny_total, shape.nz_total,
                                              delta_u, V, mu, invrho, dx1, dmx1, dx2,
                                              dmx2, Cmu, dt);
}

template <class MuArray, class RhoArray>
__global__ void couple_rhs_w_kernel(const Shape shape, real_t* rhs_w,
                                    const real_t* delta_u, const real_t* delta_v,
                                    const real_t* W, const MuArray mu,
                                    const RhoArray invrho, const real_t* dx1,
                                    const real_t* dmx1, const real_t* dx2,
                                    const real_t* dmx2, const real_t* dx3,
                                    const real_t* dmx3, const real_t Cmu,
                                    const real_t dt) {
  MPMSTD_CELL_FROM_IDX(shape);
  rhs_w[stencil::vindex(i, j, k, shape.ny_total, shape.nz_total)] -=
      stencil::couple_rhs_w_from_delta_uv_cell(i, j, k, shape.ny_total,
                                               shape.nz_total, delta_u, delta_v, W,
                                               mu, invrho, dx1, dmx1, dx2, dmx2, dx3,
                                               dmx3, Cmu, dt);
}

template <class MuArray, class RhoArray>
__global__ void couple_delta_v_kernel(const Shape shape, real_t* delta_v,
                                      const real_t* delta_w, const real_t* V,
                                      const MuArray mu, const RhoArray invrho,
                                      const real_t* dx2, const real_t* dmx2,
                                      const real_t* dx3, const real_t* dmx3,
                                      const real_t Cmu, const real_t dt) {
  MPMSTD_CELL_FROM_IDX(shape);
  delta_v[stencil::vindex(i, j, k, shape.ny_total, shape.nz_total)] -=
      stencil::couple_delta_v_from_delta_w_cell(i, j, k, shape.ny_total,
                                                shape.nz_total, delta_w, V, mu,
                                                invrho, dx2, dmx2, dx3, dmx3, Cmu, dt);
}

template <class MuArray, class RhoArray>
__global__ void couple_delta_u_kernel(const Shape shape, real_t* delta_u,
                                      const real_t* delta_v, const real_t* delta_w,
                                      const real_t* U, const MuArray mu,
                                      const RhoArray invrho, const real_t* dx1,
                                      const real_t* dmx1, const real_t* dx2,
                                      const real_t* dmx2, const real_t* dx3,
                                      const real_t* dmx3, const real_t Cmu,
                                      const real_t dt) {
  MPMSTD_CELL_FROM_IDX(shape);
  delta_u[stencil::vindex(i, j, k, shape.ny_total, shape.nz_total)] -=
      stencil::couple_delta_u_from_delta_vw_cell(i, j, k, shape.ny_total,
                                                 shape.nz_total, delta_v, delta_w, U,
                                                 mu, invrho, dx1, dmx1, dx2, dmx2,
                                                 dx3, dmx3, Cmu, dt);
}

#undef MPMSTD_CELL_FROM_IDX

}  // namespace

void couple_rhs_v_from_delta_u_variable(field::Field& rhs_v,
                                        const field::Field& delta_u,
                                        const field::Field& V,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  check_field_matches("couple_rhs_v_from_delta_u_variable", rhs_v, sub);
  check_field_matches("couple_rhs_v_from_delta_u_variable", delta_u, sub);
  const Shape shape = make_shape(sub);
  couple_rhs_v_kernel<<<shape.blocks, shape.threads>>>(
      shape, rhs_v.data(), delta_u.data(), V.data(), viscosity.data(),
      inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void couple_rhs_w_from_delta_uv_variable(field::Field& rhs_w,
                                         const field::Field& delta_u,
                                         const field::Field& delta_v,
                                         const field::Field& W,
                                         const field::Field& viscosity,
                                         const field::Field& inverse_density,
                                         const grid::Grid& grid,
                                         const parallel::Subdomain& sub,
                                         const real_t Cmu, const real_t dt) {
  check_field_matches("couple_rhs_w_from_delta_uv_variable", rhs_w, sub);
  check_field_matches("couple_rhs_w_from_delta_uv_variable", delta_u, sub);
  check_field_matches("couple_rhs_w_from_delta_uv_variable", delta_v, sub);
  const Shape shape = make_shape(sub);
  couple_rhs_w_kernel<<<shape.blocks, shape.threads>>>(
      shape, rhs_w.data(), delta_u.data(), delta_v.data(), W.data(),
      viscosity.data(), inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void block_couple_delta_v_variable(field::Field& delta_v, const field::Field& delta_w,
                                   const field::Field& V,
                                   const field::Field& viscosity,
                                   const field::Field& inverse_density,
                                   const grid::Grid& grid,
                                   const parallel::Subdomain& sub, const real_t Cmu,
                                   const real_t dt) {
  check_field_matches("block_couple_delta_v_variable", delta_v, sub);
  check_field_matches("block_couple_delta_v_variable", delta_w, sub);
  const Shape shape = make_shape(sub);
  couple_delta_v_kernel<<<shape.blocks, shape.threads>>>(
      shape, delta_v.data(), delta_w.data(), V.data(), viscosity.data(),
      inverse_density.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void block_couple_delta_u_variable(field::Field& delta_u, const field::Field& delta_v,
                                   const field::Field& delta_w, const field::Field& U,
                                   const field::Field& viscosity,
                                   const field::Field& inverse_density,
                                   const grid::Grid& grid,
                                   const parallel::Subdomain& sub, const real_t Cmu,
                                   const real_t dt) {
  check_field_matches("block_couple_delta_u_variable", delta_u, sub);
  check_field_matches("block_couple_delta_u_variable", delta_v, sub);
  check_field_matches("block_couple_delta_u_variable", delta_w, sub);
  const Shape shape = make_shape(sub);
  couple_delta_u_kernel<<<shape.blocks, shape.threads>>>(
      shape, delta_u.data(), delta_v.data(), delta_w.data(), U.data(),
      viscosity.data(), inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

// ── 상수물성 판 (plan §18-1) — UnitArray 로 같은 커널 ────────────────────────
void couple_rhs_v_from_delta_u_constant_property(
    field::Field& rhs_v, const field::Field& delta_u, const field::Field& V,
    const grid::Grid& grid, const parallel::Subdomain& sub, const real_t viscosity,
    const real_t dt) {
  check_field_matches("couple_rhs_v_from_delta_u_constant_property", rhs_v, sub);
  check_field_matches("couple_rhs_v_from_delta_u_constant_property", delta_u, sub);
  const Shape shape = make_shape(sub);
  couple_rhs_v_kernel<<<shape.blocks, shape.threads>>>(
      shape, rhs_v.data(), delta_u.data(), V.data(), equation::UnitArray{},
      equation::UnitArray{}, grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), viscosity, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void couple_rhs_w_from_delta_uv_constant_property(
    field::Field& rhs_w, const field::Field& delta_u, const field::Field& delta_v,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  check_field_matches("couple_rhs_w_from_delta_uv_constant_property", rhs_w, sub);
  check_field_matches("couple_rhs_w_from_delta_uv_constant_property", delta_u, sub);
  check_field_matches("couple_rhs_w_from_delta_uv_constant_property", delta_v, sub);
  const Shape shape = make_shape(sub);
  couple_rhs_w_kernel<<<shape.blocks, shape.threads>>>(
      shape, rhs_w.data(), delta_u.data(), delta_v.data(), W.data(),
      equation::UnitArray{}, equation::UnitArray{}, grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), viscosity, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void block_couple_delta_v_constant_property(field::Field& delta_v,
                                            const field::Field& delta_w,
                                            const field::Field& V,
                                            const grid::Grid& grid,
                                            const parallel::Subdomain& sub,
                                            const real_t viscosity, const real_t dt) {
  check_field_matches("block_couple_delta_v_constant_property", delta_v, sub);
  check_field_matches("block_couple_delta_v_constant_property", delta_w, sub);
  const Shape shape = make_shape(sub);
  couple_delta_v_kernel<<<shape.blocks, shape.threads>>>(
      shape, delta_v.data(), delta_w.data(), V.data(), equation::UnitArray{},
      equation::UnitArray{}, grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), viscosity, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void block_couple_delta_u_constant_property(field::Field& delta_u,
                                            const field::Field& delta_v,
                                            const field::Field& delta_w,
                                            const field::Field& U,
                                            const grid::Grid& grid,
                                            const parallel::Subdomain& sub,
                                            const real_t viscosity, const real_t dt) {
  check_field_matches("block_couple_delta_u_constant_property", delta_u, sub);
  check_field_matches("block_couple_delta_u_constant_property", delta_v, sub);
  check_field_matches("block_couple_delta_u_constant_property", delta_w, sub);
  const Shape shape = make_shape(sub);
  couple_delta_u_kernel<<<shape.blocks, shape.threads>>>(
      shape, delta_u.data(), delta_v.data(), delta_w.data(), U.data(),
      equation::UnitArray{}, equation::UnitArray{}, grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), viscosity, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::momentum
