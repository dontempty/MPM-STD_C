#include "equation/pressure/pencil_transpose.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

// 호스트 전치 — 양쪽 빌드 공용 (GPU 빌드도 이 .cpp 를 그대로 쓴다; solve.cpp 가 스테이징).

namespace mpmstd::equation::pressure {

namespace {

void exchange(MPI_Comm comm, const std::vector<real_t>& send, const std::vector<int>& send_counts,
              std::vector<real_t>& recv, const std::vector<int>& recv_counts) {
  const int p = static_cast<int>(send_counts.size());
  std::vector<int> sd(p, 0), rd(p, 0);
  for (int r = 1; r < p; ++r) {
    sd[r] = sd[r - 1] + send_counts[r - 1];
    rd[r] = rd[r - 1] + recv_counts[r - 1];
  }
  recv.resize(static_cast<std::size_t>(rd[p - 1] + recv_counts[p - 1]));
  MPI_Alltoallv(send.data(), send_counts.data(), sd.data(), MPI_DOUBLE, recv.data(),
                recv_counts.data(), rd.data(), MPI_DOUBLE, comm);
}

int overlap(const int a0, const int an, const int b0, const int bn, int& start) {
  start = a0 > b0 ? a0 : b0;
  const int end = (a0 + an < b0 + bn) ? a0 + an : b0 + bn;
  return end > start ? end - start : 0;
}

}  // namespace

PencilLayout::PencilLayout(const parallel::Subdomain& sub, const int n_modes) {
  const auto& topo = sub.topology();
  // 같은 z 블록끼리 — 키는 (y 블록, x 블록) 순서라 재현 가능
  MPI_Comm_split(topo.cart_comm(), topo.z().rank, topo.y().rank * topo.x().nprocs + topo.x().rank,
                 &comm);
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
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
}

PencilLayout::~PencilLayout() {
  if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
}

// ① 블록 → x 라인. 보낼 것: 내 블록의 x 전부 × (내 y ∩ 상대 y 조각) × z.  순서 (y, k, x).
void block_to_x_lines(const PencilLayout& L, const parallel::Subdomain& sub,
                      const std::vector<real_t>& field_host, const bool mirrored,
                      std::vector<real_t>& lines) {
  const int h = halo_width, nyt = sub.y().n_total, nzt = sub.z().n_total, nz = L.nz_local;
  const int bx0 = L.block_x_offset[L.rank], bxn = L.block_x_count[L.rank];
  const int by0 = L.block_y_offset[L.rank], byn = L.block_y_count[L.rank];
  std::vector<int> send_counts(L.size), recv_counts(L.size);
  std::vector<real_t> send;
  for (int r = 0; r < L.size; ++r) {
    int y0 = 0;
    const int ny = overlap(by0, byn, L.y_offset[r], L.y_count[r], y0);
    send_counts[r] = ny * nz * bxn;
    for (int y = y0; y < y0 + ny; ++y)
      for (int k = 0; k < nz; ++k)
        for (int x = bx0; x < bx0 + bxn; ++x) {
          const int i = x - bx0 + h, j = y - by0 + h;
          send.push_back(field_host[(static_cast<std::size_t>(i) * nyt + j) * nzt + (k + h)]);
        }
    int s0 = 0;
    recv_counts[r] = overlap(L.block_y_offset[r], L.block_y_count[r], L.my_y_offset(),
                             L.my_y_count(), s0) * nz * L.block_x_count[r];
  }
  std::vector<real_t> recv;
  exchange(L.comm, send, send_counts, recv, recv_counts);
  const int line = mirrored ? 2 * L.n_x : L.n_x;
  lines.assign(static_cast<std::size_t>(L.my_y_count()) * nz * line, 0.0);
  std::size_t q = 0;
  for (int r = 0; r < L.size; ++r) {
    int y0 = 0;
    const int ny = overlap(L.block_y_offset[r], L.block_y_count[r], L.my_y_offset(),
                           L.my_y_count(), y0);
    for (int y = y0; y < y0 + ny; ++y)
      for (int k = 0; k < nz; ++k)
        for (int x = L.block_x_offset[r]; x < L.block_x_offset[r] + L.block_x_count[r]; ++x) {
          real_t* row = lines.data()
                        + (static_cast<std::size_t>(y - L.my_y_offset()) * nz + k) * line;
          row[x] = recv[q];
          if (mirrored) row[line - 1 - x] = recv[q];
          ++q;
        }
  }
}

// ①⁻¹ x 라인 → 블록 (위의 정확한 역; 앞 Nx 개만 사용)
void x_lines_to_block(const PencilLayout& L, const parallel::Subdomain& sub,
                      const std::vector<real_t>& lines, const int line_length, const real_t scale,
                      const real_t subtract, std::vector<real_t>& field_host) {
  const int h = halo_width, nyt = sub.y().n_total, nzt = sub.z().n_total, nz = L.nz_local;
  std::vector<int> send_counts(L.size), recv_counts(L.size);
  std::vector<real_t> send;
  for (int r = 0; r < L.size; ++r) {
    int y0 = 0;
    const int ny = overlap(L.block_y_offset[r], L.block_y_count[r], L.my_y_offset(),
                           L.my_y_count(), y0);
    send_counts[r] = ny * nz * L.block_x_count[r];
    for (int y = y0; y < y0 + ny; ++y)
      for (int k = 0; k < nz; ++k)
        for (int x = L.block_x_offset[r]; x < L.block_x_offset[r] + L.block_x_count[r]; ++x)
          send.push_back(lines[(static_cast<std::size_t>(y - L.my_y_offset()) * nz + k) * line_length + x]
                             * scale
                         - subtract);
    int s0 = 0;
    recv_counts[r] = overlap(L.block_y_offset[L.rank], L.block_y_count[L.rank], L.y_offset[r],
                             L.y_count[r], s0) * nz * L.block_x_count[L.rank];
  }
  std::vector<real_t> recv;
  exchange(L.comm, send, send_counts, recv, recv_counts);
  const int bx0 = L.block_x_offset[L.rank], bxn = L.block_x_count[L.rank];
  const int by0 = L.block_y_offset[L.rank], byn = L.block_y_count[L.rank];
  std::size_t q = 0;
  for (int r = 0; r < L.size; ++r) {
    int y0 = 0;
    const int ny = overlap(by0, byn, L.y_offset[r], L.y_count[r], y0);
    for (int y = y0; y < y0 + ny; ++y)
      for (int k = 0; k < nz; ++k)
        for (int x = bx0; x < bx0 + bxn; ++x) {
          const int i = x - bx0 + h, j = y - by0 + h;
          field_host[(static_cast<std::size_t>(i) * nyt + j) * nzt + (k + h)] = recv[q++];
        }
  }
}

// ② x 스펙트럼 → y 스펙트럼. 보낼 것: 내 y 조각 × z × (상대 모드 조각), 순서 (jl, k, mx), 복소 2개씩.
void x_spectrum_to_y_spectrum(const PencilLayout& L, const std::vector<real_t>& sx,
                              std::vector<real_t>& sxy) {
  const int nz = L.nz_local, nm = L.n_mode_x, ny_mine = L.my_y_count();
  std::vector<int> send_counts(L.size), recv_counts(L.size);
  std::vector<real_t> send;
  send.reserve(sx.size());
  for (int r = 0; r < L.size; ++r) {
    send_counts[r] = 2 * ny_mine * nz * L.mode_count[r];
    recv_counts[r] = 2 * L.y_count[r] * nz * L.my_mode_count();
    for (int jl = 0; jl < ny_mine; ++jl)
      for (int k = 0; k < nz; ++k)
        for (int mx = L.mode_offset[r]; mx < L.mode_offset[r] + L.mode_count[r]; ++mx) {
          const std::size_t s = 2 * ((static_cast<std::size_t>(jl) * nz + k) * nm + mx);
          send.push_back(sx[s]);
          send.push_back(sx[s + 1]);
        }
  }
  std::vector<real_t> recv;
  exchange(L.comm, send, send_counts, recv, recv_counts);
  const int nml = L.my_mode_count(), m0 = L.my_mode_offset();
  sxy.assign(2 * static_cast<std::size_t>(nz) * nml * L.n_y, 0.0);
  std::size_t q = 0;
  for (int r = 0; r < L.size; ++r)
    for (int jl = 0; jl < L.y_count[r]; ++jl)
      for (int k = 0; k < nz; ++k)
        for (int mx = m0; mx < m0 + nml; ++mx) {
          const std::size_t d =
              2 * ((static_cast<std::size_t>(k) * nml + (mx - m0)) * L.n_y + (L.y_offset[r] + jl));
          sxy[d] = recv[q++];
          sxy[d + 1] = recv[q++];
        }
}

void y_spectrum_to_x_spectrum(const PencilLayout& L, const std::vector<real_t>& sxy,
                              std::vector<real_t>& sx) {
  const int nz = L.nz_local, nm = L.n_mode_x, nml = L.my_mode_count(), m0 = L.my_mode_offset();
  std::vector<int> send_counts(L.size), recv_counts(L.size);
  std::vector<real_t> send;
  send.reserve(sxy.size());
  for (int r = 0; r < L.size; ++r) {
    send_counts[r] = 2 * L.y_count[r] * nz * nml;
    recv_counts[r] = 2 * L.my_y_count() * nz * L.mode_count[r];
    for (int jl = 0; jl < L.y_count[r]; ++jl)
      for (int k = 0; k < nz; ++k)
        for (int mx = m0; mx < m0 + nml; ++mx) {
          const std::size_t s =
              2 * ((static_cast<std::size_t>(k) * nml + (mx - m0)) * L.n_y + (L.y_offset[r] + jl));
          send.push_back(sxy[s]);
          send.push_back(sxy[s + 1]);
        }
  }
  std::vector<real_t> recv;
  exchange(L.comm, send, send_counts, recv, recv_counts);
  sx.assign(2 * static_cast<std::size_t>(L.my_y_count()) * nz * nm, 0.0);
  std::size_t q = 0;
  for (int r = 0; r < L.size; ++r)
    for (int jl = 0; jl < L.my_y_count(); ++jl)
      for (int k = 0; k < nz; ++k)
        for (int mx = L.mode_offset[r]; mx < L.mode_offset[r] + L.mode_count[r]; ++mx) {
          const std::size_t d = 2 * ((static_cast<std::size_t>(jl) * nz + k) * nm + mx);
          sx[d] = recv[q++];
          sx[d + 1] = recv[q++];
        }
}

}  // namespace mpmstd::equation::pressure
