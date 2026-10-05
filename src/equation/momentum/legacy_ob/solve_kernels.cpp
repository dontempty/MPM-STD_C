#include "equation/momentum/legacy_ob/solve_kernels.hpp"

#include <cstddef>

#include "equation/momentum/legacy_ob/momentum_stencil.hpp"

// CPU 백엔드 — 커플링 수식은 momentum_stencil.hpp, 여기는 루프·산포만.

namespace mpmstd::equation::momentum {

namespace {

inline std::size_t at(const int i, const int j, const int k, const int ny_total,
                      const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

}  // namespace

void block_couple_delta_v(field::Field& dV_field, const field::Field& dW_field,
                          const field::Field& V_field, const grid::Grid& grid,
                          const parallel::Subdomain& sub, const real_t dt) {
  const int h = halo_width;
  const field::View dV = dV_field.view();
  const field::ConstView dW = dW_field.view();
  const field::ConstView V = V_field.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        dV(i, j, k) -= dt * stencil::couple_dv_from_dw(
                                dW, V, i, j, k, grid.y.cell_width.data(),
                                grid.y.center_distance.data(),
                                grid.z.center_distance.data());
}

void block_couple_delta_u(field::Field& dU_field, const field::Field& dV_field,
                          const field::Field& dW_field, const field::Field& U_field,
                          const grid::Grid& grid, const parallel::Subdomain& sub,
                          const real_t dt) {
  const int h = halo_width;
  const field::View dU = dU_field.view();
  const field::ConstView dV = dV_field.view();
  const field::ConstView dW = dW_field.view();
  const field::ConstView U = U_field.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        dU(i, j, k) -= dt * stencil::couple_du_from_dv_dw(
                                dV, dW, U, i, j, k, grid.x.cell_width.data(),
                                grid.x.center_distance.data(),
                                grid.y.center_distance.data(),
                                grid.z.center_distance.data());
}

}  // namespace mpmstd::equation::momentum
