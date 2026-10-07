#include <cmath>
#include <limits>

#include "post/wall_richardson.hpp"

// CPU 백엔드 — 열별 |Ri_f| 와 가중 보수 (GPU 쌍둥이 wall_richardson_kernels.cu).

namespace mpmstd::post {

void local_flux_richardson(std::vector<real_t>& richardson, const field::Field& U,
                           const field::Field& T, const grid::Grid& grid,
                           const parallel::Subdomain& sub, const real_t nu, const real_t alpha,
                           const int side, const real_t z_plus_reference) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int i2 = sub.y().n_local;
  const int k_first = side == 0 ? h : nz - h - 1;       // 벽에 붙은 첫 내부 셀
  const int k_ghost = side == 0 ? h - 1 : nz - h;       // 그 벽의 ghost
  const real_t gap = grid.z.center[k_first] - grid.z.center[k_ghost];  // 부호 포함 (cold 는 음수)
  const field::ConstView u = U.view();
  const field::ConstView t = T.view();
  richardson.assign(static_cast<std::size_t>(sub.x().n_local) * i2, 0);
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j) {
      const real_t dudz = (u(i, j, k_first) - u(i, j, k_ghost)) / gap;
      const real_t dtdz = (t(i, j, k_first) - t(i, j, k_ghost)) / gap;
      const real_t u_tau_sq = nu * std::abs(dudz);           // u_τ²
      const real_t heat_flux = alpha * std::abs(dtdz);        // f_w
      const real_t denominator = u_tau_sq * u_tau_sq;         // u_τ⁴
      richardson[static_cast<std::size_t>(i - h) * i2 + (j - h)] =
          denominator > real_t(0) ? z_plus_reference * nu * heat_flux / denominator
                                  : std::numeric_limits<real_t>::infinity();
    }
}

void complement_weights(field::Buffer& other, const field::Buffer& indicator) {
  if (other.size() != indicator.size()) other.assign(indicator.size());
  real_t* o = other.data();
  const real_t* w = indicator.data();
  for (std::size_t c = 0; c < indicator.size(); ++c) o[c] = 1 - w[c];
}

}  // namespace mpmstd::post
