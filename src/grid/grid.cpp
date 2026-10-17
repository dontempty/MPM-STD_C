#include "grid/grid.hpp"

#include <stdexcept>

namespace mpmstd::grid {

namespace {

[[noreturn]] void fail(const std::string& axis_name, const std::string& what) {
  throw std::invalid_argument("grid axis " + axis_name + ": " + what);
}

}  // namespace

Axis1D make_axis(const std::vector<real_t>& gf, AxisRange r, const std::string& axis_name) {
  // ── 입력 검증 ──
  if (r.n_global <= 0) fail(axis_name, "n_global must be > 0, got " + std::to_string(r.n_global));
  if (gf.size() != static_cast<std::size_t>(r.n_global) + 1)
    fail(axis_name, "global_faces must have n_global+1 = " + std::to_string(r.n_global + 1)
                        + " entries, got " + std::to_string(gf.size()));
  for (int i = 0; i < r.n_global; ++i)
    if (!(gf[i] < gf[i + 1]))
      fail(axis_name, "faces must be strictly increasing (face[" + std::to_string(i + 1)
                          + "] <= face[" + std::to_string(i) + "])");
  if (r.n_local <= 0) fail(axis_name, "n_local must be > 0, got " + std::to_string(r.n_local));
  if (r.offset < 0 || r.offset + r.n_local > r.n_global)
    fail(axis_name, "range [" + std::to_string(r.offset) + ", "
                        + std::to_string(r.offset + r.n_local) + ") is outside [0, "
                        + std::to_string(r.n_global) + ")");

  Axis1D axis;
  axis.n_global = r.n_global;
  axis.offset = r.offset;
  axis.n_local = r.n_local;
  axis.n_total = r.n_local + 2 * halo_width;
  axis.length = gf.back() - gf.front();

  const int n_glb = r.n_global;
  const int n_tot = axis.n_total;

  // 전역 면 좌표 조회 — 전역 범위 밖(ghost 면)은 끝 셀 폭의 거울 반사.
  const auto face_global = [&gf, n_glb](int g) -> real_t {
    if (g < 0) return 2 * gf[0] - gf[-g];                     // 하단 밖
    if (g > n_glb) return 2 * gf[n_glb] - gf[2 * n_glb - g];  // 상단 밖
    return gf[g];
  };

  // 1) 로컬 면: 로컬 면 k ↔ 전역 면 offset - halo_width + k
  axis.face.resize(n_tot + 1);
  for (int k = 0; k <= n_tot; ++k) axis.face[k] = face_global(r.offset - halo_width + k);

  // 2) 셀 폭, 3) 셀 중심, 4) 중심 간 거리
  axis.cell_width.resize(n_tot);
  axis.center.resize(n_tot);
  axis.center_distance.resize(n_tot);
  for (int i = 0; i < n_tot; ++i) {
    axis.cell_width[i] = axis.face[i + 1] - axis.face[i];
    axis.center[i] = real_t{0.5} * (axis.face[i] + axis.face[i + 1]);
  }
  axis.center_distance[0] = axis.cell_width[0];  // center[-1] 이 없다 — PaScaL 외삽 관례
  for (int i = 1; i < n_tot; ++i)
    axis.center_distance[i] = axis.center[i] - axis.center[i - 1];

  axis.inverse_cell_width.resize(n_tot);
  axis.inverse_center_distance.resize(n_tot);
  for (int i = 0; i < n_tot; ++i) {
    axis.inverse_cell_width[i] = real_t{1} / axis.cell_width[i];
    axis.inverse_center_distance[i] = real_t{1} / axis.center_distance[i];
  }

#ifdef MPMSTD_GPU
  axis.device_cell_width.assign(axis.cell_width.size());
  axis.device_cell_width.upload(axis.cell_width);
  axis.device_center_distance.assign(axis.center_distance.size());
  axis.device_center_distance.upload(axis.center_distance);
  axis.device_inverse_cell_width.assign(axis.inverse_cell_width.size());
  axis.device_inverse_cell_width.upload(axis.inverse_cell_width);
  axis.device_inverse_center_distance.assign(axis.inverse_center_distance.size());
  axis.device_inverse_center_distance.upload(axis.inverse_center_distance);
#endif
  return axis;
}

}  // namespace mpmstd::grid
