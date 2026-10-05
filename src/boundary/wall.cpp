#include "boundary/wall.hpp"

#include <stdexcept>
#include <string>

namespace mpmstd::boundary {

namespace {

// 필드와 분할이 같은 격자를 보고 있는지 — 잘못된 (필드, 분할) 짝을 조기에 잡는다.
void check_dims(const char* who, const field::Field& f, const parallel::Subdomain& sub) {
  if (f.nx_total() != sub.x().n_total || f.ny_total() != sub.y().n_total
      || f.nz_total() != sub.z().n_total)
    throw std::invalid_argument(
        std::string(who) + ": field '" + f.name() + "' has extents "
        + std::to_string(f.nx_total()) + "x" + std::to_string(f.ny_total()) + "x"
        + std::to_string(f.nz_total()) + " but the subdomain expects "
        + std::to_string(sub.x().n_total) + "x" + std::to_string(sub.y().n_total)
        + "x" + std::to_string(sub.z().n_total));
}

void check_axis(const char* who, const int axis) {
  if (axis < 0 || axis > 2)
    throw std::invalid_argument(std::string(who) + ": axis must be 0, 1 or 2");
}

// 축 axis 의 슬롯 평면 전체(다른 두 축의 ghost 칸 포함)를 순회 — 모서리 값도
// 일관되게 정의되어 스텐실 확장 시 놀랄 일이 없다. fn(i, j, k) 를 부른다.
// (루프 본문은 Field 대신 POD 뷰를 쓴다 — GPU 쌍둥이(wall.cu)와 같은 모양)
template <class Fn>
void for_each_in_plane(const field::Field& f, const int axis, const int slot, Fn&& fn) {
  const int n[3] = {f.nx_total(), f.ny_total(), f.nz_total()};
  const int p_axis = (axis == 0) ? 1 : 0;        // 평면을 이루는 두 축
  const int q_axis = (axis == 2) ? 1 : 2;
  int idx[3];
  idx[axis] = slot;
  for (int p = 0; p < n[p_axis]; ++p) {
    idx[p_axis] = p;
    for (int q = 0; q < n[q_axis]; ++q) {
      idx[q_axis] = q;
      fn(idx[0], idx[1], idx[2]);
    }
  }
}

// 축 axis 의 minus/plus 벽을 이 랭크가 소유하는가 (이웃 == MPI_PROC_NULL 하나로 판정)
bool owns_minus_wall(const parallel::Subdomain& sub, const int axis) {
  return sub.topology().axis(axis).minus_neighbor == MPI_PROC_NULL;
}
bool owns_plus_wall(const parallel::Subdomain& sub, const int axis) {
  return sub.topology().axis(axis).plus_neighbor == MPI_PROC_NULL;
}

// 축 axis 의 총 길이
int n_total(const field::Field& f, const int axis) {
  return axis == 0 ? f.nx_total() : (axis == 1 ? f.ny_total() : f.nz_total());
}

// 슬롯 along axis 를 d 만큼 옮긴 인덱스 — ghost 슬롯에서 첫 내부 슬롯으로
inline void shift(int& i, int& j, int& k, const int axis, const int d) {
  if (axis == 0) i += d; else if (axis == 1) j += d; else k += d;
}

}  // namespace

// halo_width = 1 전제 (common/types.hpp): minus ghost 평면 = 슬롯 0, 첫 내부 평면 =
// 슬롯 halo_width; plus ghost 평면 = n_total − 1, 마지막 내부 = n_total − 1 − halo_width.

namespace {

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
  const int ghost = side == 0 ? 0 : n_total(f, axis) - 1;
  const int to_interior = side == 0 ? +h : -h;
  const field::View v = f.view();
  for_each_in_plane(f, axis, ghost, [&](int i, int j, int k) {
    int ii = i, jj = j, kk = k;
    shift(ii, jj, kk, axis, to_interior);
    v(i, j, k) = 2 * value - v(ii, jj, kk);
  });
}

void fill_wall_value(field::Field& f, const parallel::Subdomain& sub, const int axis,
                     const int side, const real_t value) {
  check_dims("fill_wall_value", f, sub);
  check_axis("fill_wall_value", axis);
  check_side("fill_wall_value", side);
  if (!owns_wall(sub, axis, side)) return;
  const int ghost = side == 0 ? 0 : n_total(f, axis) - 1;
  const field::View v = f.view();
  for_each_in_plane(f, axis, ghost, [&](int i, int j, int k) { v(i, j, k) = value; });
}

void fill_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub,
                             const int axis, const int side) {
  check_dims("fill_wall_zero_gradient", f, sub);
  check_axis("fill_wall_zero_gradient", axis);
  check_side("fill_wall_zero_gradient", side);
  if (!owns_wall(sub, axis, side)) return;
  const int h = halo_width;
  const int ghost = side == 0 ? 0 : n_total(f, axis) - 1;
  const int to_interior = side == 0 ? +h : -h;
  const field::View v = f.view();
  for_each_in_plane(f, axis, ghost, [&](int i, int j, int k) {
    int ii = i, jj = j, kk = k;
    shift(ii, jj, kk, axis, to_interior);
    v(i, j, k) = v(ii, jj, kk);
  });
}

void zero_wall_faces(field::Field& f, const parallel::Subdomain& sub, const int axis,
                     const int side) {
  check_dims("zero_wall_faces", f, sub);
  check_axis("zero_wall_faces", axis);
  check_side("zero_wall_faces", side);
  if (!owns_wall(sub, axis, side)) return;
  const int h = halo_width;
  // minus 벽면 = "내부" 슬롯 h (conventions.md §3), plus 벽면 = ghost 슬롯 n − h
  const int face = side == 0 ? h : n_total(f, axis) - h;
  const field::View v = f.view();
  for_each_in_plane(f, axis, face, [&](int i, int j, int k) { v(i, j, k) = 0; });
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
