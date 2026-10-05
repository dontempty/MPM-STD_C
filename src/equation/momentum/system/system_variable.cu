#include "equation/momentum/system/system_variable.hpp"

#include <cstddef>

#include "common/cuda_check.hpp"
#include "equation/momentum/momentum_variable_stencil.hpp"

// GPU 백엔드 — system_variable.cpp 의 쌍둥이. 수식은 variable_sweep_band 한 곳.

namespace mpmstd::equation::momentum {

namespace {

// sweep_axis: 0/1/2 — 밴드 슬롯 매핑과 이웃 선택만 다르다 (CPU 루프와 동일).
template <class MuArray, class RhoArray>
__global__ void build_kernel(const int sweep_axis, real_t* A, real_t* B, real_t* C,
                             real_t* D, const real_t* source, const real_t* q,
                             const real_t* advecting_velocity, const MuArray viscosity,
                             const RhoArray inverse_density, const real_t* dx_along,
                             const real_t* dmx_along, const real_t* dx_own,
                             const real_t* dmx_own, const long long own_stride,
                             const int own_axis, const int h, const int i1,
                             const int i2, const int i3, const int ny_total,
                             const int nz_total, const real_t Cmu, const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;

  int ii, jj, kk;
  std::size_t p;
  if (sweep_axis == 0) {
    kk = static_cast<int>(idx % i3);
    jj = static_cast<int>((idx / i3) % i2);
    ii = static_cast<int>(idx / (static_cast<long long>(i3) * i2));
    p = static_cast<std::size_t>(ii) * (i2 * i3) + jj * i3 + kk;
  } else if (sweep_axis == 1) {
    kk = static_cast<int>(idx % i3);
    ii = static_cast<int>((idx / i3) % i1);
    jj = static_cast<int>(idx / (static_cast<long long>(i3) * i1));
    p = static_cast<std::size_t>(jj) * (i1 * i3) + ii * i3 + kk;
  } else {
    jj = static_cast<int>(idx % i2);
    ii = static_cast<int>((idx / i2) % i1);
    kk = static_cast<int>(idx / (static_cast<long long>(i2) * i1));
    p = static_cast<std::size_t>(kk) * (i1 * i2) + ii * i2 + jj;
  }
  const int i = ii + h, j = jj + h, k = kk + h;
  const long long gi = stencil::vindex(i, j, k, ny_total, nz_total);
  long long gm, gp;
  int a;
  if (sweep_axis == 0) {
    gm = stencil::vindex(i - 1, j, k, ny_total, nz_total);
    gp = stencil::vindex(i + 1, j, k, ny_total, nz_total);
    a = i;
  } else if (sweep_axis == 1) {
    gm = stencil::vindex(i, j - 1, k, ny_total, nz_total);
    gp = stencil::vindex(i, j + 1, k, ny_total, nz_total);
    a = j;
  } else {
    gm = stencil::vindex(i, j, k - 1, ny_total, nz_total);
    gp = stencil::vindex(i, j, k + 1, ny_total, nz_total);
    a = k;
  }
  const int own_index = (own_axis == 0) ? i : (own_axis == 1) ? j : k;
  stencil::variable_sweep_band(own_axis == sweep_axis, false, gi, gm, gp, a,
                               own_index, own_stride, q, advecting_velocity,
                               viscosity, inverse_density, dx_along, dmx_along,
                               dx_own, dmx_own, Cmu, dt, A[p], B[p], C[p]);
  D[p] = source[gi];
}

struct DeviceAxes {
  const real_t* dx;
  const real_t* dmx;
};

DeviceAxes device_axes(const int axis, const grid::Grid& grid) {
  if (axis == 0)
    return {grid.x.device_cell_width.data(), grid.x.device_center_distance.data()};
  if (axis == 1)
    return {grid.y.device_cell_width.data(), grid.y.device_center_distance.data()};
  return {grid.z.device_cell_width.data(), grid.z.device_center_distance.data()};
}

template <class MuArray, class RhoArray>
void launch(const int sweep_axis, real_t* A, real_t* B, real_t* C, real_t* D,
            const real_t* source, const real_t* q, const real_t* advecting_velocity,
            const MuArray viscosity, const RhoArray inverse_density,
            const grid::Grid& grid, const parallel::Subdomain& sub, const int own_axis,
            const real_t Cmu, const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  const DeviceAxes along = device_axes(sweep_axis, grid);
  const DeviceAxes own = device_axes(own_axis, grid);
  const long long own_stride =
      (own_axis == 0) ? static_cast<long long>(sub.y().n_total) * sub.z().n_total
      : (own_axis == 1) ? static_cast<long long>(sub.z().n_total)
                        : 1;
  build_kernel<<<blocks, threads>>>(sweep_axis, A, B, C, D, source, q,
                                    advecting_velocity, viscosity, inverse_density,
                                    along.dx, along.dmx, own.dx, own.dmx, own_stride,
                                    own_axis, halo_width, i1, i2, i3,
                                    sub.y().n_total, sub.z().n_total, Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void build_x_sweep_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  launch(0, A, B, C, D, source, q, advecting_velocity, viscosity, inverse_density,
         grid, sub, own_axis, Cmu, dt);
}

void build_y_sweep_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  launch(1, A, B, C, D, source, q, advecting_velocity, viscosity, inverse_density,
         grid, sub, own_axis, Cmu, dt);
}

void build_z_sweep_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  launch(2, A, B, C, D, source, q, advecting_velocity, viscosity, inverse_density,
         grid, sub, own_axis, Cmu, dt);
}

// ── 상수물성 판 (plan §18-1) — UnitArray 로 같은 커널을 돈다 ──────────────────
void build_x_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub,
                                     const int own_axis, const real_t viscosity,
                                     const real_t dt) {
  launch(0, A, B, C, D, source, q, advecting_velocity, equation::UnitArray{},
         equation::UnitArray{}, grid, sub, own_axis, viscosity, dt);
}

void build_y_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub,
                                     const int own_axis, const real_t viscosity,
                                     const real_t dt) {
  launch(1, A, B, C, D, source, q, advecting_velocity, equation::UnitArray{},
         equation::UnitArray{}, grid, sub, own_axis, viscosity, dt);
}

void build_z_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub,
                                     const int own_axis, const real_t viscosity,
                                     const real_t dt) {
  launch(2, A, B, C, D, source, q, advecting_velocity, equation::UnitArray{},
         equation::UnitArray{}, grid, sub, own_axis, viscosity, dt);
}

}  // namespace mpmstd::equation::momentum
