#include "boundary/open.hpp"

#include <mpi.h>

#include <stdexcept>

namespace mpmstd::boundary {

namespace {
bool owns_minus(const parallel::Subdomain& sub, int axis) {
  return sub.topology().axis(axis).minus_neighbor == MPI_PROC_NULL;
}
bool owns_plus(const parallel::Subdomain& sub, int axis) {
  return sub.topology().axis(axis).plus_neighbor == MPI_PROC_NULL;
}
}  // namespace

InflowPlane InflowPlane::from_profile(const std::vector<real_t>& u_z, const std::vector<real_t>& v_z,
                                      const std::vector<real_t>& w_z, const int ny) {
  InflowPlane p;
  p.ny_total = ny;
  p.nz_total = static_cast<int>(u_z.size());
  const std::size_t n = static_cast<std::size_t>(ny) * p.nz_total;
  p.u.resize(n); p.v.resize(n); p.w.resize(n);
  for (int j = 0; j < ny; ++j)
    for (int k = 0; k < p.nz_total; ++k) {
      p.at(p.u, j, k) = u_z[k];
      p.at(p.v, j, k) = v_z[k];
      p.at(p.w, j, k) = w_z[k];
    }
  return p;
}

void set_inflow(field::Field& U, field::Field& V, field::Field& W, const InflowPlane& p,
                const VelocityGhost ghost, const parallel::Subdomain& sub) {
  if (!owns_minus(sub, 0)) return;
  const int h = halo_width;
  const int ny = sub.y().n_total, nz = sub.z().n_total;
  if (p.ny_total != ny || p.nz_total != nz)
    throw std::invalid_argument("set_inflow: plane size must be ny_total x nz_total");
  const field::View u = U.view(), v = V.view(), w = W.view();
  for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) {
      const std::size_t q = static_cast<std::size_t>(j) * nz + k;
      u(h, j, k) = p.u[q];  // 유입면 = 내부 슬롯 h (conventions §3)
      if (ghost == VelocityGhost::Mirror) {
        v(0, j, k) = 2 * p.v[q] - v(h, j, k);
        w(0, j, k) = 2 * p.w[q] - w(h, j, k);
      } else {
        v(0, j, k) = p.v[q];
        w(0, j, k) = p.w[q];
      }
    }
}

void extract_x_plane(const field::Field& F, const int i, std::vector<real_t>& out,
                     const parallel::Subdomain& sub) {
  const int ny = sub.y().n_total, nz = sub.z().n_total;
  out.resize(static_cast<std::size_t>(ny) * nz);
  const field::ConstView f = F.view();
  for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) out[static_cast<std::size_t>(j) * nz + k] = f(i, j, k);
}

void update_outflow_convective(field::Field& U, field::Field& V, field::Field& W,
                               const real_t convective_velocity, const real_t dt,
                               const grid::Grid& grid, const parallel::Subdomain& sub) {
  if (!owns_plus(sub, 0)) return;
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int i_face = nx - h;   // 유출 면 (ghost 슬롯)
  const int i_ghost = nx - 1;  // 셀중심 성분의 ghost
  const field::View u = U.view(), v = V.view(), w = W.view();
  const real_t cu = convective_velocity * dt / grid.x.cell_width[i_face - 1];
  const real_t cc = convective_velocity * dt / grid.x.center_distance[i_ghost];
  for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k) {
      u(i_face, j, k) -= cu * (u(i_face, j, k) - u(i_face - 1, j, k));
      v(i_ghost, j, k) -= cc * (v(i_ghost, j, k) - v(i_ghost - 1, j, k));
      w(i_ghost, j, k) -= cc * (w(i_ghost, j, k) - w(i_ghost - 1, j, k));
    }
}

void set_top_normal_from_continuity(field::Field& W, const field::Field& U,
                                    const field::Field& V, const grid::Grid& grid,
                                    const parallel::Subdomain& sub) {
  if (!owns_plus(sub, 2)) return;
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int k_face = nz - h, k_cell = k_face - 1;
  const field::View w = W.view();
  const field::ConstView u = U.view(), v = V.view();
  const real_t dz = grid.z.cell_width[k_cell];
  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j) {
      const real_t dudx = (u(i + 1, j, k_cell) - u(i, j, k_cell)) / grid.x.cell_width[i];
      const real_t dvdy = (v(i, j + 1, k_cell) - v(i, j, k_cell)) / grid.y.cell_width[j];
      w(i, j, k_face) = w(i, j, k_cell) - dz * (dudx + dvdy);
    }
}

void set_top_normal_profile(field::Field& W, const std::vector<real_t>& w_top,
                            const parallel::Subdomain& sub) {
  if (!owns_plus(sub, 2)) return;
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  if (static_cast<int>(w_top.size()) != nx)
    throw std::invalid_argument("set_top_normal_profile: profile size must be nx_total");
  const field::View w = W.view();
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) w(i, j, nz - h) = w_top[i];
}

real_t correct_outflow_flux(field::Field& U, const field::Field& W, const grid::Grid& grid,
                            const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const field::View u = U.view();
  const field::ConstView w = W.view();
  // 로컬 합: [0] 유입, [1] 상단 유출, [2] 유출면, [3] 유출면 면적
  real_t local[4] = {0, 0, 0, 0};
  if (owns_minus(sub, 0))
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k)
        local[0] += u(h, j, k) * grid.y.cell_width[j] * grid.z.cell_width[k];
  if (owns_plus(sub, 2))
    for (int i = h; i < nx - h; ++i)
      for (int j = h; j < ny - h; ++j)
        local[1] += w(i, j, nz - h) * grid.x.cell_width[i] * grid.y.cell_width[j];
  if (owns_plus(sub, 0))
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) {
        const real_t area = grid.y.cell_width[j] * grid.z.cell_width[k];
        local[2] += u(nx - h, j, k) * area;
        local[3] += area;
      }
  real_t global[4];
  MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  const real_t correction = (global[0] - global[1] - global[2]) / global[3];
  if (owns_plus(sub, 0))
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k) u(nx - h, j, k) += correction;
  return correction;
}

}  // namespace mpmstd::boundary
