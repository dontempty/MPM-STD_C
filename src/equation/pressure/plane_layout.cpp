#include "equation/pressure/pencil_transpose.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// 평면 묶음 레이아웃 + 전치 인덱스 표 (호스트, 양쪽 빌드 공용) — pencil_transpose.hpp 의 PlaneLayout.
// 인덱스 표는 CPU 백엔드와 GPU 의 MPMSTD_PENCIL_PACK=index 대조 경로가 쓴다 (GPU 기본은 구조화 커널).

namespace mpmstd::equation::pressure {

namespace {

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

bool PlaneLayout::usable(const parallel::Subdomain& sub) {
  const auto& topo = sub.topology();
  const int groups = topo.x().nprocs * topo.y().nprocs;
  if (groups <= 1) return false;
  // PaScaL 분산 풀이는 랭크당 2 행 이상 — 가장 적게 받는 그룹 랭크 = floor(nz_local / P)
  int ok = sub.z().n_local / groups >= 2 ? 1 : 0, all = 0;
  MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, topo.cart_comm());
  return all == 1;
}

PlaneLayout::PlaneLayout(const parallel::Subdomain& sub, const bool mirrored_lines) {
  const auto& topo = sub.topology();
  const int group_key = topo.y().rank * topo.x().nprocs + topo.x().rank;
  MPI_Comm_split(topo.cart_comm(), topo.z().rank, group_key, &comm);
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_split(topo.cart_comm(), 0, topo.z().rank * size + rank, &z_comm);
  const char* mode = std::getenv("PASCAL_TDMA_MPI_MODE");
  host_staging = mode != nullptr && std::strcmp(mode, "host") == 0;

  n_x = sub.x().n_global;
  n_y = sub.y().n_global;
  nz_local = sub.z().n_local;
  halo = halo_width;
  ny_total = sub.y().n_total;
  nz_total = sub.z().n_total;
  line = mirrored_lines ? 2 * n_x : n_x;
  mirrored = mirrored_lines;
  for (int g = 0; g < size; ++g) {
    const parallel::ParaRange ks = parallel::split_evenly(nz_local, size, g);
    plane_offset.push_back(ks.offset);
    plane_count.push_back(ks.count);
  }
  owns_bottom = topo.z().minus_neighbor == MPI_PROC_NULL && rank == 0;
  owns_top = topo.z().plus_neighbor == MPI_PROC_NULL && rank == size - 1;

  int mine[4] = {sub.x().offset, sub.x().n_local, sub.y().offset, sub.y().n_local};
  std::vector<int> all(4 * static_cast<std::size_t>(size));
  MPI_Allgather(mine, 4, MPI_INT, all.data(), 4, MPI_INT, comm);
  for (int r = 0; r < size; ++r) {
    block_x_offset.push_back(all[4 * r]);
    block_x_count.push_back(all[4 * r + 1]);
    block_y_offset.push_back(all[4 * r + 2]);
    block_y_count.push_back(all[4 * r + 3]);
  }

  const int h = halo, nyt = ny_total, nzt = nz_total;
  const int bx0 = block_x_offset[rank], bxn = block_x_count[rank];
  const int by0 = block_y_offset[rank], byn = block_y_count[rank];
  const int k0 = my_plane_offset(), nzp = my_plane_count();
  const auto field_at = [&](int x, int y, int k) {  // 내 블록의 전역 (x, y), 로컬 z 행 k → 장 배열 번호
    return static_cast<int>((static_cast<std::size_t>(x - bx0 + h) * nyt + (y - by0 + h)) * nzt + (k + h));
  };
  const auto line_at = [&](int y, int kk, int x) {  // 내 평면 묶음의 x 라인 번호 (y 전역, kk 내 평면)
    return static_cast<int>((static_cast<std::size_t>(y) * nzp + kk) * line + x);
  };
  std::vector<int> send_n(size), recv_n(size);

  // ① 블록 → 평면: g 에게 내 블록 (y, g 의 평면, x), 순서 (y, kk, x) / r 에게서 r 블록 (y, 내 평면, x)
  {
    PencilExchange& e = block_to_planes;
    e.width = 1;
    for (int g = 0; g < size; ++g) {
      send_n[g] = byn * plane_count[g] * bxn;
      for (int y = by0; y < by0 + byn; ++y)
        for (int kk = 0; kk < plane_count[g]; ++kk)
          for (int x = bx0; x < bx0 + bxn; ++x) e.send_index.push_back(field_at(x, y, plane_offset[g] + kk));
      recv_n[g] = block_y_count[g] * nzp * block_x_count[g];
      for (int y = block_y_offset[g]; y < block_y_offset[g] + block_y_count[g]; ++y)
        for (int kk = 0; kk < nzp; ++kk)
          for (int x = block_x_offset[g]; x < block_x_offset[g] + block_x_count[g]; ++x) {
            e.recv_index.push_back(line_at(y, kk, x));
            if (mirrored_lines) e.recv_mirror.push_back(line_at(y, kk, line - 1 - x));
          }
    }
    finish_counts(e, send_n, recv_n);
  }
  // ①⁻¹ 평면 → 블록: 위의 정확한 역 (앞 Nx 개만)
  {
    PencilExchange& e = planes_to_block;
    e.width = 1;
    for (int r = 0; r < size; ++r) {
      send_n[r] = block_y_count[r] * nzp * block_x_count[r];
      for (int y = block_y_offset[r]; y < block_y_offset[r] + block_y_count[r]; ++y)
        for (int kk = 0; kk < nzp; ++kk)
          for (int x = block_x_offset[r]; x < block_x_offset[r] + block_x_count[r]; ++x)
            e.send_index.push_back(line_at(y, kk, x));
      recv_n[r] = byn * plane_count[r] * bxn;
      for (int y = by0; y < by0 + byn; ++y)
        for (int kk = 0; kk < plane_count[r]; ++kk)
          for (int x = bx0; x < bx0 + bxn; ++x) e.recv_index.push_back(field_at(x, y, plane_offset[r] + kk));
    }
    finish_counts(e, send_n, recv_n);
  }
  (void)k0;
  block_to_planes.kind = PencilKind::block_to_planes;
  planes_to_block.kind = PencilKind::planes_to_block;
  for (PencilExchange* e : {&block_to_planes, &planes_to_block}) {
    e->planes = this;
    prepare_exchange(*e);
  }
}

PlaneLayout::~PlaneLayout() {
  if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
  if (z_comm != MPI_COMM_NULL) MPI_Comm_free(&z_comm);
}

}  // namespace mpmstd::equation::pressure
