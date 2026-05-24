#include "grid/grid.hpp"

#include <stdexcept>

namespace mpmstd::grid {

Grid::Grid(const parallel::mpi::Subdomain& sub,
           std::array<AxisConfig, 3>       axes)
  : sub_(sub), axes_(axes) {

  // Validate.
  for (int a = 0; a < 3; ++a) {
    if (axes_[a].n_global <= 0)
      throw std::invalid_argument("Grid: n_global must be > 0 on every axis");
    if (axes_[a].n_global != sub_.n_global()[a])
      throw std::invalid_argument(
        "Grid: axis n_global does not match Subdomain's n_global");
  }

  for (int a = 0; a < 3; ++a) {
    build_axis_(a);
  }
}

void Grid::build_axis_(int axis) {
  const int  n_tot  = sub_.n_total()[axis];
  const int  offset = sub_.global_offset()[axis];
  const int  n_glb  = axes_[axis].n_global;
  const real_t L    = axes_[axis].length;

  // 1) Generate the global face coordinates first, then slice this rank's part.
  //    Global faces live on [0, n_glb], so global_faces has size n_glb + 1.
  auto global_faces = make_face_coordinates(axes_[axis].stretch,
                                              n_glb, L, axes_[axis].gamma);

  // 2) Local face coordinates with halos.
  //    Interior face indices on this rank: [offset, offset + n_int] (n_int + 1 entries).
  //    With halo width 1, we also need one face below and one above.
  //    Total local face count = n_tot + 1.
  auto& xf = xf_[axis];
  xf.resize(static_cast<std::size_t>(n_tot) + 1);

  auto face_global = [&](int g) -> real_t {
    // Clamp / extrapolate at outer ends.
    if (g < 0)              return 2.0 * global_faces[0]     - global_faces[1];
    if (g > n_glb)          return 2.0 * global_faces[n_glb] - global_faces[n_glb - 1];
    return global_faces[g];
  };

  // Local face[k] corresponds to global face index (offset - kHaloWidth + k).
  for (int k = 0; k <= n_tot; ++k) {
    int g = offset - kHaloWidth + k;
    xf[k] = face_global(g);
  }

  // 3) Cell widths dx[i] = xf[i+1] - xf[i]   (size n_tot)
  auto& dx = dx_[axis];
  dx.resize(static_cast<std::size_t>(n_tot));
  for (int i = 0; i < n_tot; ++i) {
    dx[i] = xf[i + 1] - xf[i];
  }

  // 4) Cell-center coordinates xc[i] = 0.5*(xf[i] + xf[i+1])   (size n_tot)
  auto& xc = xc_[axis];
  xc.resize(static_cast<std::size_t>(n_tot));
  for (int i = 0; i < n_tot; ++i) {
    xc[i] = 0.5 * (xf[i] + xf[i + 1]);
  }

  // 5) Face-to-face distance dmx[i] = xc[i] - xc[i-1] for i >= 1,
  //    and dmx[0] = dx[0] (extrapolated convention matching PaScaL_TCS).
  auto& dmx = dmx_[axis];
  dmx.resize(static_cast<std::size_t>(n_tot));
  dmx[0] = dx[0];
  for (int i = 1; i < n_tot; ++i) {
    dmx[i] = xc[i] - xc[i - 1];
  }
}

} // namespace mpmstd::grid
