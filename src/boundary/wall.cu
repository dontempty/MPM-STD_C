#include "boundary/wall.hpp"

#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

// GPU 백엔드 — wall.cpp 의 쌍둥이. ghost 공식은 한 줄짜리라 양쪽에 그대로 쓰고
// 상호 참조 주석으로 짝을 맞춘다 (wall.hpp 의 공식 표가 정본).
// 축 매개변수화: 한 평면 = 축 axis 의 슬롯 하나, 스레드는 다른 두 축 (p, q) 를 편다.

namespace mpmstd::boundary {

namespace {

enum : int { kThreads = 256 };

void check_dims(const char* who, const field::Field& f, const parallel::Subdomain& sub) {
  if (f.nx_total() != sub.x().n_total || f.ny_total() != sub.y().n_total
      || f.nz_total() != sub.z().n_total)
    throw std::invalid_argument(std::string(who) + ": field '" + f.name()
                                + "' does not match the subdomain");
}

void check_axis(const char* who, const int axis) {
  if (axis < 0 || axis > 2)
    throw std::invalid_argument(std::string(who) + ": axis must be 0, 1 or 2");
}

// 평면 기하: 축 axis 의 슬롯 slot, 평면 축 (p_axis, q_axis) 의 길이 (np, nq).
struct Plane {
  int axis, slot, p_axis, q_axis, np, nq;
};

Plane make_plane(const field::Field& f, const int axis, const int slot) {
  const int n[3] = {f.nx_total(), f.ny_total(), f.nz_total()};
  Plane pl;
  pl.axis = axis;
  pl.slot = slot;
  pl.p_axis = (axis == 0) ? 1 : 0;
  pl.q_axis = (axis == 2) ? 1 : 2;
  pl.np = n[pl.p_axis];
  pl.nq = n[pl.q_axis];
  return pl;
}

// 스레드 idx → (i, j, k): 평면 좌표 (p, q) + 축 슬롯 (slot + shift)
__device__ inline void decode(const Plane pl, const int idx, const int shift, int& i,
                              int& j, int& k) {
  const int p = idx / pl.nq, q = idx % pl.nq;
  int v[3];
  v[pl.axis] = pl.slot + shift;
  v[pl.p_axis] = p;
  v[pl.q_axis] = q;
  i = v[0];
  j = v[1];
  k = v[2];
}

__global__ void mirror_kernel(const field::View v, const Plane pl, const int to_interior,
                              const real_t value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pl.np * pl.nq) return;
  int i, j, k, ii, jj, kk;
  decode(pl, idx, 0, i, j, k);
  decode(pl, idx, to_interior, ii, jj, kk);
  v(i, j, k) = 2 * value - v(ii, jj, kk);  // = wall.cpp 의 mirror
}

__global__ void value_kernel(const field::View v, const Plane pl, const real_t value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pl.np * pl.nq) return;
  int i, j, k;
  decode(pl, idx, 0, i, j, k);
  v(i, j, k) = value;
}

__global__ void copy_kernel(const field::View v, const Plane pl, const int to_interior) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pl.np * pl.nq) return;
  int i, j, k, ii, jj, kk;
  decode(pl, idx, 0, i, j, k);
  decode(pl, idx, to_interior, ii, jj, kk);
  v(i, j, k) = v(ii, jj, kk);  // zero-gradient
}

inline int blocks_for(const int n) { return (n + kThreads - 1) / kThreads; }

bool owns_minus_wall(const parallel::Subdomain& sub, const int axis) {
  return sub.topology().axis(axis).minus_neighbor == MPI_PROC_NULL;
}
bool owns_plus_wall(const parallel::Subdomain& sub, const int axis) {
  return sub.topology().axis(axis).plus_neighbor == MPI_PROC_NULL;
}
int n_total(const field::Field& f, const int axis) {
  return axis == 0 ? f.nx_total() : (axis == 1 ? f.ny_total() : f.nz_total());
}

bool owns_wall(const parallel::Subdomain& sub, const int axis, const int side) {
  return side == 0 ? owns_minus_wall(sub, axis) : owns_plus_wall(sub, axis);
}

void check_side(const char* who, const int side) {
  if (side != 0 && side != 1)
    throw std::invalid_argument(std::string(who) + ": side must be 0 (minus) or 1 (plus)");
}

}  // namespace

void fill_wall_mirror(field::Field& f, const parallel::Subdomain& sub, const int axis,
                      const int side, const real_t value) {
  check_dims("fill_wall_mirror", f, sub);
  check_axis("fill_wall_mirror", axis);
  check_side("fill_wall_mirror", side);
  if (!owns_wall(sub, axis, side)) return;
  const int h = halo_width;
  const Plane pl = make_plane(f, axis, side == 0 ? 0 : n_total(f, axis) - 1);
  mirror_kernel<<<blocks_for(pl.np * pl.nq), kThreads>>>(f.view(), pl, side == 0 ? +h : -h,
                                                         value);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void fill_wall_value(field::Field& f, const parallel::Subdomain& sub, const int axis,
                     const int side, const real_t value) {
  check_dims("fill_wall_value", f, sub);
  check_axis("fill_wall_value", axis);
  check_side("fill_wall_value", side);
  if (!owns_wall(sub, axis, side)) return;
  const Plane pl = make_plane(f, axis, side == 0 ? 0 : n_total(f, axis) - 1);
  value_kernel<<<blocks_for(pl.np * pl.nq), kThreads>>>(f.view(), pl, value);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void fill_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub,
                             const int axis, const int side) {
  check_dims("fill_wall_zero_gradient", f, sub);
  check_axis("fill_wall_zero_gradient", axis);
  check_side("fill_wall_zero_gradient", side);
  if (!owns_wall(sub, axis, side)) return;
  const int h = halo_width;
  const Plane pl = make_plane(f, axis, side == 0 ? 0 : n_total(f, axis) - 1);
  copy_kernel<<<blocks_for(pl.np * pl.nq), kThreads>>>(f.view(), pl, side == 0 ? +h : -h);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void zero_wall_faces(field::Field& f, const parallel::Subdomain& sub, const int axis,
                     const int side) {
  check_dims("zero_wall_faces", f, sub);
  check_axis("zero_wall_faces", axis);
  check_side("zero_wall_faces", side);
  if (!owns_wall(sub, axis, side)) return;
  const int h = halo_width;
  // minus 벽면 = 내부 슬롯 h, plus 벽면 = ghost 슬롯 n − h
  const Plane pl = make_plane(f, axis, side == 0 ? h : n_total(f, axis) - h);
  value_kernel<<<blocks_for(pl.np * pl.nq), kThreads>>>(f.view(), pl, 0);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

// ── z 전용 이름 = axis 2, minus → plus 순 (원래 구현과 같은 순서) ─────────────

void fill_z_wall_mirror(field::Field& f, const parallel::Subdomain& sub,
                        const real_t value_bottom, const real_t value_top) {
  fill_wall_mirror(f, sub, 2, 0, value_bottom);
  fill_wall_mirror(f, sub, 2, 1, value_top);
}

void fill_z_wall_value(field::Field& f, const parallel::Subdomain& sub,
                       const real_t value_bottom, const real_t value_top) {
  fill_wall_value(f, sub, 2, 0, value_bottom);
  fill_wall_value(f, sub, 2, 1, value_top);
}

void fill_z_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub) {
  fill_wall_zero_gradient(f, sub, 2, 0);
  fill_wall_zero_gradient(f, sub, 2, 1);
}

void zero_z_wall_faces(field::Field& w, const parallel::Subdomain& sub) {
  zero_wall_faces(w, sub, 2, 0);
  zero_wall_faces(w, sub, 2, 1);
}

}  // namespace mpmstd::boundary
