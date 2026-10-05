#include "parallel/subdomain/subdomain.hpp"



#include <stdexcept>

namespace mpmstd::parallel {

ParaRange split_evenly(int n_global, int nprocs, int rank) {
  if (nprocs <= 0 || rank < 0 || rank >= nprocs)
    throw std::invalid_argument("split_evenly: invalid nprocs " + std::to_string(nprocs)
                                + " / rank " + std::to_string(rank));
  if (n_global < nprocs)
    throw std::invalid_argument("split_evenly: cannot give every rank at least one cell ("
                                + std::to_string(n_global) + " cells < "
                                + std::to_string(nprocs) + " ranks)");

  // PaScaL_TDMA_C 의 para_range 와 동일: 나머지는 앞쪽 랭크부터 1 개씩.
  const int base = n_global / nprocs;
  const int remainder = n_global % nprocs;
  ParaRange r;
  r.offset = rank * base + (rank < remainder ? rank : remainder);
  r.count = base + (rank < remainder ? 1 : 0);
  return r;
}

Subdomain::Subdomain(const Topology& topo, int nx_global, int ny_global, int nz_global)
    : topo_(topo) {
  {
    const ParaRange r = split_evenly(nx_global, topo.x().nprocs, topo.x().rank);
    x_ = {nx_global, r.offset, r.count, r.count + 2 * halo_width};
  }
  {
    const ParaRange r = split_evenly(ny_global, topo.y().nprocs, topo.y().rank);
    y_ = {ny_global, r.offset, r.count, r.count + 2 * halo_width};
  }
  {
    const ParaRange r = split_evenly(nz_global, topo.z().nprocs, topo.z().rank);
    z_ = {nz_global, r.offset, r.count, r.count + 2 * halo_width};
  }

  // halo 평면 datatype — real_t == double (common/types.hpp; 바뀌면 여기도 같이).
  // 배열 모양 {x.n_total, y.n_total, z.n_total}, 선형 인덱스 (i*ny + j)*nz + k.
  // starts 는 전부 0 으로 두고, 평면 위치는 교환할 때 포인터를 밀어서 고른다.

  {  // y-평면: j 를 고정한 한 장 = {nx, 1, nz}
    int sizes[3] = {x_.n_total, y_.n_total, z_.n_total};
    int subsizes[3] = {x_.n_total, 1, z_.n_total};
    int starts[3] = {0, 0, 0};
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE,
                             &y_plane_type_);
    MPI_Type_commit(&y_plane_type_);
  }
  {  // z-평면: k 를 고정한 한 장 = {nx, ny, 1} — 원소 간격이 nz 라 가장 비싸다
    int sizes[3] = {x_.n_total, y_.n_total, z_.n_total};
    int subsizes[3] = {x_.n_total, y_.n_total, 1};
    int starts[3] = {0, 0, 0};
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE,
                             &z_plane_type_);
    MPI_Type_commit(&z_plane_type_);
  
#ifdef MPMSTD_GPU
  const std::size_t y_face = static_cast<std::size_t>(x_.n_total) * z_.n_total;
  const std::size_t z_face = static_cast<std::size_t>(x_.n_total) * y_.n_total;
  y_send_minus_.assign(y_face);
  y_send_plus_.assign(y_face);
  y_recv_minus_.assign(y_face);
  y_recv_plus_.assign(y_face);
  z_send_minus_.assign(z_face);
  z_send_plus_.assign(z_face);
  z_recv_minus_.assign(z_face);
  z_recv_plus_.assign(z_face);
#endif
}
}

Subdomain::~Subdomain() {
  if (z_plane_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&z_plane_type_);
  if (y_plane_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&y_plane_type_);
}

std::size_t Subdomain::n_elements() const {
  return static_cast<std::size_t>(x_.n_total) * y_.n_total * z_.n_total;
}

void Subdomain::exchange_halo(real_t* f) const {
#ifdef MPMSTD_GPU
  // 디바이스 경로 — 팩 커널 + 연속 버퍼 MPI (subdomain_halo.cu; 헤더 주석 참조)
  exchange_halo_device(f);
  return;
#endif
  const int nx = x_.n_total;
  const int ny = y_.n_total;
  const int nz = z_.n_total;
  const int h = halo_width;

  // ── x 방향 ────────────────────────────────────────────────────────────
  // x-평면 한 장(i 고정)은 연속 ny*nz 개 → datatype 없이 개수로 보낸다.
  {
    const int plane = ny * nz;
    // 첫 내부 평면(i = h) → −이웃.   +이웃에게서 받아 → 상단 ghost(i = nx-h)
    MPI_Sendrecv(f + h * plane,        plane, MPI_DOUBLE, topo_.x().minus_neighbor, 100,
                 f + (nx - h) * plane, plane, MPI_DOUBLE, topo_.x().plus_neighbor,  100,
                 topo_.x().comm, MPI_STATUS_IGNORE);
    // 끝 내부 평면(i = nx-2h) → +이웃.  −이웃에게서 받아 → 하단 ghost(i = 0)
    MPI_Sendrecv(f + (nx - 2 * h) * plane, plane, MPI_DOUBLE, topo_.x().plus_neighbor,  101,
                 f,                         plane, MPI_DOUBLE, topo_.x().minus_neighbor, 101,
                 topo_.x().comm, MPI_STATUS_IGNORE);
  }

  // ── y 방향 ────────────────────────────────────────────────────────────
  // y-평면(j 고정)은 벌어져 있다 → subarray datatype. 평면 j 의 시작 = f + j*nz.
  {
    MPI_Sendrecv(f + h * nz,        1, y_plane_type_, topo_.y().minus_neighbor, 200,
                 f + (ny - h) * nz, 1, y_plane_type_, topo_.y().plus_neighbor,  200,
                 topo_.y().comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(f + (ny - 2 * h) * nz, 1, y_plane_type_, topo_.y().plus_neighbor,  201,
                 f,                      1, y_plane_type_, topo_.y().minus_neighbor, 201,
                 topo_.y().comm, MPI_STATUS_IGNORE);
  }

  // ── z 방향 ────────────────────────────────────────────────────────────
  // z-평면(k 고정) — 평면 k 의 시작 = f + k. 벽에서는 이웃이 MPI_PROC_NULL
  // 이라 Sendrecv 가 자동 no-op 이다 (분기문이 필요 없다).
  {
    MPI_Sendrecv(f + h,        1, z_plane_type_, topo_.z().minus_neighbor, 300,
                 f + (nz - h), 1, z_plane_type_, topo_.z().plus_neighbor,  300,
                 topo_.z().comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(f + (nz - 2 * h), 1, z_plane_type_, topo_.z().plus_neighbor,  301,
                 f,                 1, z_plane_type_, topo_.z().minus_neighbor, 301,
                 topo_.z().comm, MPI_STATUS_IGNORE);
  }
}

}  // namespace mpmstd::parallel
