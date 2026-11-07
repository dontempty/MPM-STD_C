#include "block_cyclic_exchange.hpp"

#include <climits>
#include <stdexcept>

#include "device_loop.hpp"
#include "exchange.hpp"

namespace fft_test {

namespace {

#define EXCHANGE_BEGIN(name) \
  do { if (timer) timer->begin(name); } while (0)
#define EXCHANGE_END(name) \
  do { if (timer) timer->end(name); } while (0)

// 구간 [offset, offset + count) 에서 v ≡ residue (mod q) 인 첫 값과 개수
void congruent_range(const int offset, const int count, const int residue, const int q, int& first, int& n) {
  first = offset + ((residue - offset) % q + q) % q;
  n = first < offset + count ? (offset + count - 1 - first) / q + 1 : 0;
}

// 행 복사: dst[row·dst_stride + c] = src[row·src_stride + c] · scale − subtract
struct CopyRows {
  real_t* dst;
  long long dst_stride;
  const real_t* src;
  long long src_stride;
  int length;
  real_t scale, subtract;
  MPMSTD_HD void operator()(const long long n) const {
    const long long row = n / length;
    const int c = static_cast<int>(n - row * length);
    dst[row * dst_stride + c] = src[row * src_stride + c] * scale - subtract;
  }
};

#ifdef MPMSTD_GPU

// 장은 k 가 연속, 버퍼는 x 가 연속 → 32×32 공유메모리 타일로 양쪽 연속 접근 (pencil_exchange.cu 와 같은 방식).
constexpr int kTile = 32;
constexpr int kTileRows = 8;

// 버퍼 [(i·nz + z)·bxn + xl] = 장 (xl, y0 + i·q, z).  그리드 (⌈bxn/32⌉, ⌈nz/32⌉, 평면 수)
__global__ void field_to_rows_kernel(real_t* buffer, const real_t* field, const int h, const int nyt, const int nzt,
                                     const int y0, const int q, const int bxn, const int nz) {
  __shared__ real_t tile[kTile][kTile + 1];
  const int i = static_cast<int>(blockIdx.z);
  const int x0 = static_cast<int>(blockIdx.x) * kTile, z0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  const int y = y0 + i * q;
  for (int p = 0; p < kTile; p += kTileRows) {   // 읽기: tx 가 z 를 걷는다
    const int xl = x0 + ty + p, z = z0 + tx;
    if (xl < bxn && z < nz) tile[ty + p][tx] = field[(static_cast<std::size_t>(xl + h) * nyt + (y + h)) * nzt + (z + h)];
  }
  __syncthreads();
  for (int p = 0; p < kTile; p += kTileRows) {   // 쓰기: tx 가 x 를 걷는다
    const int xl = x0 + tx, z = z0 + ty + p;
    if (xl < bxn && z < nz) buffer[(static_cast<std::size_t>(i) * nz + z) * bxn + xl] = tile[tx][ty + p];
  }
}

// 역: 장 (xl, y0 + i·q, z) = 버퍼 [(i·nz + z)·bxn + xl] · scale − subtract.  그리드 (⌈nz/32⌉, ⌈bxn/32⌉, 평면 수)
__global__ void rows_to_field_kernel(real_t* field, const real_t* buffer, const int h, const int nyt, const int nzt,
                                     const int y0, const int q, const int bxn, const int nz, const real_t scale,
                                     const real_t subtract) {
  __shared__ real_t tile[kTile][kTile + 1];
  const int i = static_cast<int>(blockIdx.z);
  const int z0 = static_cast<int>(blockIdx.x) * kTile, x0 = static_cast<int>(blockIdx.y) * kTile;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  const int y = y0 + i * q;
  for (int p = 0; p < kTile; p += kTileRows) {   // 읽기: tx 가 x 를 걷는다
    const int z = z0 + ty + p, xl = x0 + tx;
    if (xl < bxn && z < nz) tile[ty + p][tx] = buffer[(static_cast<std::size_t>(i) * nz + z) * bxn + xl];
  }
  __syncthreads();
  for (int p = 0; p < kTile; p += kTileRows) {   // 쓰기: tx 가 z 를 걷는다
    const int z = z0 + tx, xl = x0 + ty + p;
    if (xl < bxn && z < nz)
      field[(static_cast<std::size_t>(xl + h) * nyt + (y + h)) * nzt + (z + h)] = tile[tx][ty + p] * scale - subtract;
  }
}

#endif

void field_to_rows(real_t* buffer, const real_t* field, const int nyt, const int nzt, const int y0, const int q,
                   const int plane_count, const int bxn, const int nz) {
  if (plane_count == 0 || bxn == 0) return;
  const int h = mpmstd::halo_width;
#ifdef MPMSTD_GPU
  const dim3 threads(kTile, kTileRows);
  const dim3 blocks((bxn + kTile - 1) / kTile, (nz + kTile - 1) / kTile, plane_count);
  field_to_rows_kernel<<<blocks, threads>>>(buffer, field, h, nyt, nzt, y0, q, bxn, nz);
  MPMSTD_CUDA_CHECK_LAUNCH();
#else
  for (int i = 0; i < plane_count; ++i)
    for (int z = 0; z < nz; ++z)
      for (int xl = 0; xl < bxn; ++xl)
        buffer[(static_cast<std::size_t>(i) * nz + z) * bxn + xl] =
            field[(static_cast<std::size_t>(xl + h) * nyt + (y0 + i * q + h)) * nzt + (z + h)];
#endif
}

void rows_to_field(real_t* field, const real_t* buffer, const int nyt, const int nzt, const int y0, const int q,
                   const int plane_count, const int bxn, const int nz, const real_t scale, const real_t subtract) {
  if (plane_count == 0 || bxn == 0) return;
  const int h = mpmstd::halo_width;
#ifdef MPMSTD_GPU
  const dim3 threads(kTile, kTileRows);
  const dim3 blocks((nz + kTile - 1) / kTile, (bxn + kTile - 1) / kTile, plane_count);
  rows_to_field_kernel<<<blocks, threads>>>(field, buffer, h, nyt, nzt, y0, q, bxn, nz, scale, subtract);
  MPMSTD_CUDA_CHECK_LAUNCH();
#else
  for (int i = 0; i < plane_count; ++i)
    for (int z = 0; z < nz; ++z)
      for (int xl = 0; xl < bxn; ++xl)
        field[(static_cast<std::size_t>(xl + h) * nyt + (y0 + i * q + h)) * nzt + (z + h)] =
            buffer[(static_cast<std::size_t>(i) * nz + z) * bxn + xl] * scale - subtract;
#endif
}

}  // namespace

BlockCyclicExchange::BlockCyclicExchange(const mpmstd::parallel::Subdomain& sub, const CyclicLayout& layout,
                                         const bool host_staging)
    : layout_(layout),
      host_staging_(host_staging),
      ny_total_(sub.y().n_total),
      nz_total_(sub.z().n_total),
      block_x_count_(sub.x().n_local) {
  const CyclicDims& d = layout.dims();
  const int p = d.q;
  int mine[4] = {sub.x().offset, sub.x().n_local, sub.y().offset, sub.y().n_local};
  std::vector<int> all(4 * static_cast<std::size_t>(p));
  MPI_Allgather(mine, 4, MPI_INT, all.data(), 4, MPI_INT, layout.group());

  first_local_y_.assign(p, 0);
  plane_count_.assign(p, 0);
  peer_x_offset_.assign(p, 0);
  peer_x_count_.assign(p, 0);
  peer_first_k_.assign(p, 0);
  peer_plane_count_.assign(p, 0);
  send_counts_.assign(p, 0);
  send_displs_.assign(p, 0);
  recv_counts_.assign(p, 0);
  recv_displs_.assign(p, 0);

  long long send_total = 0, recv_total = 0;
  for (int g = 0; g < p; ++g) {
    int first = 0, n = 0;
    congruent_range(mine[2], mine[3], g, d.q, first, n);   // 내 블록에서 g 몫의 y
    first_local_y_[g] = first - mine[2];
    plane_count_[g] = n;
    const long long sending = static_cast<long long>(n) * d.nz * mine[1];
    send_displs_[g] = static_cast<int>(send_total);
    send_counts_[g] = static_cast<int>(sending);
    send_total += sending;

    congruent_range(all[4 * g + 2], all[4 * g + 3], d.s, d.q, first, n);   // g 의 블록에서 내 몫의 y
    peer_x_offset_[g] = all[4 * g];
    peer_x_count_[g] = all[4 * g + 1];
    peer_first_k_[g] = first / d.q;
    peer_plane_count_[g] = n;
    const long long receiving = static_cast<long long>(n) * d.nz * all[4 * g + 1];
    recv_displs_[g] = static_cast<int>(recv_total);
    recv_counts_[g] = static_cast<int>(receiving);
    recv_total += receiving;
    if (send_total > INT_MAX || recv_total > INT_MAX)
      throw std::invalid_argument("BlockCyclicExchange: MPI count or displacement exceeds int");
  }
  if (recv_total != d.line_elements())
    throw std::logic_error("BlockCyclicExchange: received cell count != cyclic line size");
}

void BlockCyclicExchange::to_cyclic(const mpmstd::field::Field& rhs, real_t* lines, real_t* block_scratch,
                                    real_t* cyclic_scratch, mpmstd::common::PhaseTimer* timer) {
  const CyclicDims& d = layout_.dims();
  EXCHANGE_BEGIN("fftu.block_to_cyclic");
  if (d.q == 1) {
    field_to_rows(lines, rhs.data(), ny_total_, nz_total_, 0, 1, plane_count_[0], block_x_count_, d.nz);
    EXCHANGE_END("fftu.block_to_cyclic");
    return;
  }
  for (int g = 0; g < d.q; ++g)
    field_to_rows(block_scratch + send_displs_[g], rhs.data(), ny_total_, nz_total_, first_local_y_[g], d.q,
                  plane_count_[g], block_x_count_, d.nz);
  EXCHANGE_BEGIN("fftu.block_to_cyclic_mpi");
  alltoallv_backend(block_scratch, send_counts_, send_displs_, cyclic_scratch, recv_counts_, recv_displs_,
                    layout_.group(), host_staging_);
  EXCHANGE_END("fftu.block_to_cyclic_mpi");
  for (int r = 0; r < d.q; ++r) {
    const long long rows = static_cast<long long>(peer_plane_count_[r]) * d.nz;
    for_each(rows * peer_x_count_[r],
             CopyRows{lines + static_cast<std::size_t>(peer_first_k_[r]) * d.nz * d.n_x + peer_x_offset_[r], d.n_x,
                      cyclic_scratch + recv_displs_[r], peer_x_count_[r], peer_x_count_[r], 1, 0});
  }
  EXCHANGE_END("fftu.block_to_cyclic");
}

void BlockCyclicExchange::to_block(const real_t* lines, mpmstd::field::Field& phi, const real_t scale,
                                   const real_t subtract, real_t* block_scratch, real_t* cyclic_scratch,
                                   mpmstd::common::PhaseTimer* timer) {
  const CyclicDims& d = layout_.dims();
  EXCHANGE_BEGIN("fftu.cyclic_to_block");
  if (d.q == 1) {
    rows_to_field(phi.data(), lines, ny_total_, nz_total_, 0, 1, plane_count_[0], block_x_count_, d.nz, scale, subtract);
    EXCHANGE_END("fftu.cyclic_to_block");
    return;
  }
  for (int r = 0; r < d.q; ++r) {
    const long long rows = static_cast<long long>(peer_plane_count_[r]) * d.nz;
    for_each(rows * peer_x_count_[r],
             CopyRows{cyclic_scratch + recv_displs_[r], peer_x_count_[r],
                      lines + static_cast<std::size_t>(peer_first_k_[r]) * d.nz * d.n_x + peer_x_offset_[r], d.n_x,
                      peer_x_count_[r], scale, subtract});
  }
  EXCHANGE_BEGIN("fftu.cyclic_to_block_mpi");
  alltoallv_backend(cyclic_scratch, recv_counts_, recv_displs_, block_scratch, send_counts_, send_displs_,
                    layout_.group(), host_staging_);
  EXCHANGE_END("fftu.cyclic_to_block_mpi");
  for (int g = 0; g < d.q; ++g)
    rows_to_field(phi.data(), block_scratch + send_displs_[g], ny_total_, nz_total_, first_local_y_[g], d.q,
                  plane_count_[g], block_x_count_, d.nz, 1, 0);
  EXCHANGE_END("fftu.cyclic_to_block");
}

}  // namespace fft_test
