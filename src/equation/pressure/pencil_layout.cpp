#include "equation/pressure/pencil_transpose.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// 펜슬 레이아웃 + 전치 인덱스 표 (호스트, 양쪽 빌드 공용). 실제 팩·통신·언팩은 pencil_exchange.{cpp,cu}.

namespace mpmstd::equation::pressure {

namespace {

int overlap(const int a0, const int an, const int b0, const int bn, int& start) {
  start = a0 > b0 ? a0 : b0;
  const int end = (a0 + an < b0 + bn) ? a0 + an : b0 + bn;
  return end > start ? end - start : 0;
}

void finish_counts(PencilExchange& e, const std::vector<int>& send_elems,
                   const std::vector<int>& recv_elems) {
  const int p = static_cast<int>(send_elems.size());
  e.send_counts.assign(p, 0); e.send_displs.assign(p, 0);
  e.recv_counts.assign(p, 0); e.recv_displs.assign(p, 0);
  for (int r = 0; r < p; ++r) {
    e.send_counts[r] = send_elems[r] * e.width;
    e.recv_counts[r] = recv_elems[r] * e.width;
    if (r > 0) {
      e.send_displs[r] = e.send_displs[r - 1] + e.send_counts[r - 1];
      e.recv_displs[r] = e.recv_displs[r - 1] + e.recv_counts[r - 1];
    }
  }
}

}  // namespace

PencilExchange::~PencilExchange() { release_exchange(*this); }

PencilLayout::PencilLayout(const parallel::Subdomain& sub, const int n_modes,
                           const bool mirrored_lines) {
  const auto& topo = sub.topology();
  // 같은 z 블록끼리 — 키는 (y 블록, x 블록) 순서라 재현 가능
  MPI_Comm_split(topo.cart_comm(), topo.z().rank, topo.y().rank * topo.x().nprocs + topo.x().rank,
                 &comm);
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  const char* mode = std::getenv("PASCAL_TDMA_MPI_MODE");
  host_staging = mode != nullptr && std::strcmp(mode, "host") == 0;
  n_x = sub.x().n_global;
  n_y = sub.y().n_global;
  nz_local = sub.z().n_local;
  n_mode_x = n_modes;
  if (n_y < size || n_mode_x < size)
    throw std::invalid_argument("pressure pencil: ny (" + std::to_string(n_y) + ") and x modes ("
                                + std::to_string(n_mode_x) + ") must be >= npx*npy ("
                                + std::to_string(size) + ")");
  int mine[4] = {sub.x().offset, sub.x().n_local, sub.y().offset, sub.y().n_local};
  std::vector<int> all(4 * static_cast<std::size_t>(size));
  MPI_Allgather(mine, 4, MPI_INT, all.data(), 4, MPI_INT, comm);
  for (int r = 0; r < size; ++r) {
    block_x_offset.push_back(all[4 * r]);
    block_x_count.push_back(all[4 * r + 1]);
    block_y_offset.push_back(all[4 * r + 2]);
    block_y_count.push_back(all[4 * r + 3]);
    const parallel::ParaRange ys = parallel::split_evenly(n_y, size, r);
    const parallel::ParaRange ms = parallel::split_evenly(n_mode_x, size, r);
    y_offset.push_back(ys.offset);
    y_count.push_back(ys.count);
    mode_offset.push_back(ms.offset);
    mode_count.push_back(ms.count);
  }

  const int h = halo_width, nyt = sub.y().n_total, nzt = sub.z().n_total, nz = nz_local;
  const int bx0 = block_x_offset[rank], bxn = block_x_count[rank];
  const int by0 = block_y_offset[rank], byn = block_y_count[rank];
  const int line = mirrored_lines ? 2 * n_x : n_x;
  const auto field_at = [&](int x, int y, int k) {  // 내 블록의 전역 (x, y) → 장 배열 번호
    return static_cast<int>((static_cast<std::size_t>(x - bx0 + h) * nyt + (y - by0 + h)) * nzt + (k + h));
  };
  const auto line_at = [&](int y, int k, int x) {    // 내 y 조각의 x 라인 번호
    return static_cast<int>((static_cast<std::size_t>(y - my_y_offset()) * nz + k) * line + x);
  };
  std::vector<int> send_n(size), recv_n(size);

  // ① 블록 → x 라인: 보낼 것 = 내 블록 x 전부 × (내 y ∩ 상대 y 조각) × z, 순서 (y, k, x)
  {
    PencilExchange& e = block_to_lines;
    e.width = 1;
    for (int r = 0; r < size; ++r) {
      int y0 = 0;
      const int ny = overlap(by0, byn, y_offset[r], y_count[r], y0);
      send_n[r] = ny * nz * bxn;
      for (int y = y0; y < y0 + ny; ++y)
        for (int k = 0; k < nz; ++k)
          for (int x = bx0; x < bx0 + bxn; ++x) e.send_index.push_back(field_at(x, y, k));
      int s0 = 0;
      const int ry = overlap(block_y_offset[r], block_y_count[r], my_y_offset(), my_y_count(), s0);
      recv_n[r] = ry * nz * block_x_count[r];
      for (int y = s0; y < s0 + ry; ++y)
        for (int k = 0; k < nz; ++k)
          for (int x = block_x_offset[r]; x < block_x_offset[r] + block_x_count[r]; ++x) {
            e.recv_index.push_back(line_at(y, k, x));
            if (mirrored_lines) e.recv_mirror.push_back(line_at(y, k, line - 1 - x));
          }
    }
    finish_counts(e, send_n, recv_n);
  }
  // ①⁻¹ x 라인 → 블록: 위의 정확한 역 (앞 Nx 개만)
  {
    PencilExchange& e = lines_to_block;
    e.width = 1;
    for (int r = 0; r < size; ++r) {
      int y0 = 0;
      const int ny = overlap(block_y_offset[r], block_y_count[r], my_y_offset(), my_y_count(), y0);
      send_n[r] = ny * nz * block_x_count[r];
      for (int y = y0; y < y0 + ny; ++y)
        for (int k = 0; k < nz; ++k)
          for (int x = block_x_offset[r]; x < block_x_offset[r] + block_x_count[r]; ++x)
            e.send_index.push_back(line_at(y, k, x));
      int s0 = 0;
      const int ry = overlap(by0, byn, y_offset[r], y_count[r], s0);
      recv_n[r] = ry * nz * bxn;
      for (int y = s0; y < s0 + ry; ++y)
        for (int k = 0; k < nz; ++k)
          for (int x = bx0; x < bx0 + bxn; ++x) e.recv_index.push_back(field_at(x, y, k));
    }
    finish_counts(e, send_n, recv_n);
  }
  // ② x 스펙트럼 → y 스펙트럼 (복소): 보낼 것 = 내 y 조각 × z × 상대 모드 조각, 순서 (jl, k, mx)
  const int nm = n_mode_x, nml = my_mode_count(), m0 = my_mode_offset(), ny_mine = my_y_count();
  {
    PencilExchange& e = x_to_y_spectrum;
    e.width = 2;
    for (int r = 0; r < size; ++r) {
      send_n[r] = ny_mine * nz * mode_count[r];
      for (int jl = 0; jl < ny_mine; ++jl)
        for (int k = 0; k < nz; ++k)
          for (int mx = mode_offset[r]; mx < mode_offset[r] + mode_count[r]; ++mx)
            e.send_index.push_back((jl * nz + k) * nm + mx);
      recv_n[r] = y_count[r] * nz * nml;
      for (int jl = 0; jl < y_count[r]; ++jl)
        for (int k = 0; k < nz; ++k)
          for (int mx = m0; mx < m0 + nml; ++mx)
            e.recv_index.push_back((k * nml + (mx - m0)) * n_y + (y_offset[r] + jl));
    }
    finish_counts(e, send_n, recv_n);
  }
  // ②⁻¹ y 스펙트럼 → x 스펙트럼
  {
    PencilExchange& e = y_to_x_spectrum;
    e.width = 2;
    for (int r = 0; r < size; ++r) {
      send_n[r] = y_count[r] * nz * nml;
      for (int jl = 0; jl < y_count[r]; ++jl)
        for (int k = 0; k < nz; ++k)
          for (int mx = m0; mx < m0 + nml; ++mx)
            e.send_index.push_back((k * nml + (mx - m0)) * n_y + (y_offset[r] + jl));
      recv_n[r] = ny_mine * nz * mode_count[r];
      for (int jl = 0; jl < ny_mine; ++jl)
        for (int k = 0; k < nz; ++k)
          for (int mx = mode_offset[r]; mx < mode_offset[r] + mode_count[r]; ++mx)
            e.recv_index.push_back((jl * nz + k) * nm + mx);
    }
    finish_counts(e, send_n, recv_n);
  }
  for (PencilExchange* e : {&block_to_lines, &lines_to_block, &x_to_y_spectrum, &y_to_x_spectrum})
    prepare_exchange(*e);
}

PencilLayout::~PencilLayout() {
  if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
}

}  // namespace mpmstd::equation::pressure
