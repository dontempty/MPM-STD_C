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

}  // namespace

// 아래 전부: ghost "평면 전체"(다른 축의 ghost 칸 포함)를 채운다 — 모서리 값도
// 일관되게 정의되어 스텐실 확장 시 놀랄 일이 없다. halo_width = 1 전제
// (common/types.hpp): 하단 ghost 평면 k = 0, 첫 내부 평면 k = halo_width.
// (루프 본문은 Field 대신 POD 뷰를 쓴다 — GPU 쌍둥이(wall.cu)와 같은 모양)

void fill_z_wall_mirror(field::Field& f, const parallel::Subdomain& sub,
                        real_t value_bottom, real_t value_top) {
  check_dims("fill_z_wall_mirror", f, sub);
  const int nx = f.nx_total();
  const int ny = f.ny_total();
  const int nz = f.nz_total();
  const int h = halo_width;
  const field::View v = f.view();

  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {   // 하단 벽 소유
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, 0) = 2 * value_bottom - v(i, j, h);
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {    // 상단 벽 소유
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, nz - 1) = 2 * value_top - v(i, j, nz - 1 - h);
  }
}

void fill_z_wall_value(field::Field& f, const parallel::Subdomain& sub,
                       real_t value_bottom, real_t value_top) {
  check_dims("fill_z_wall_value", f, sub);
  const int nx = f.nx_total();
  const int ny = f.ny_total();
  const int nz = f.nz_total();
  const field::View v = f.view();

  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, 0) = value_bottom;
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, nz - 1) = value_top;
  }
}

void fill_z_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub) {
  check_dims("fill_z_wall_zero_gradient", f, sub);
  const int nx = f.nx_total();
  const int ny = f.ny_total();
  const int nz = f.nz_total();
  const int h = halo_width;
  const field::View v = f.view();

  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, 0) = v(i, j, h);
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, nz - 1) = v(i, j, nz - 1 - h);
  }
}

void zero_z_wall_faces(field::Field& w, const parallel::Subdomain& sub) {
  check_dims("zero_z_wall_faces", w, sub);
  const int nx = w.nx_total();
  const int ny = w.ny_total();
  const int nz = w.nz_total();
  const int h = halo_width;
  const field::View v = w.view();

  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, h) = 0;          // 하단 벽면 — "내부" 슬롯이다 (conventions.md §3)
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        v(i, j, nz - h) = 0;     // 상단 벽면 — ghost 슬롯이다
  }
}

}  // namespace mpmstd::boundary
