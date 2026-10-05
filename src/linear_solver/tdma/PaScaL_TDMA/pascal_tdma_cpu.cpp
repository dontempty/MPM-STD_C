#include "linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cpu.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "parallel/subdomain/subdomain.hpp"  // split_evenly — 격자와 같은 분배 규칙

// 아래 모든 MPI 호출이 MPI_DOUBLE 을 쓴다. real_t 를 바꾸는 날 여기도 같이.
static_assert(std::is_same<mpmstd::real_t, double>::value,
              "tdma_cpu uses MPI_DOUBLE; change it together with real_t");

namespace mpmstd::pascal_tdma {

namespace {

// 레이아웃 정의 그 자체 (헤더 참조): sys 가 빠르고 row 가 느리다.
inline std::size_t idx2(const int sys, const int row, const int nsys) {
  return static_cast<std::size_t>(sys) + static_cast<std::size_t>(row) * nsys;
}

}  // namespace

// ── 보통 Thomas ──────────────────────────────────────────────────────────────
// row 는 순차 의존이라 바깥(호스트) 루프, sys 는 독립이라 안쪽 병렬 루프다 —
// GPU 에서는 row 마다 커널 하나가 되고, 인접 스레드(sys)가 연속 메모리를 읽어
// 자연히 코얼레싱된다 (PaScaL_TDMA CUDA 커널과 같은 병렬 모양).
// b 는 수정하지 않지만 계약은 "네 배열 모두 파괴"로 통일한다 (cyclic 은 b 도
// 고치므로).
void thomas_many(real_t* a, real_t* b, real_t* c, real_t* d,
                 const int nsys, const int nrow) {
  if (nrow == 1) {
    for (int s = 0; s < nsys; ++s) d[s] /= b[s];
    return;
  }

  for (int s = 0; s < nsys; ++s) {
    const real_t inv = 1 / b[s];
    c[s] *= inv;
    d[s] *= inv;
  }
  for (int row = 1; row < nrow; ++row) {
    for (int s = 0; s < nsys; ++s) {
      const real_t r = 1 / (b[idx2(s, row, nsys)]
                            - a[idx2(s, row, nsys)] * c[idx2(s, row - 1, nsys)]);
      d[idx2(s, row, nsys)] = r * (d[idx2(s, row, nsys)]
                                   - a[idx2(s, row, nsys)] * d[idx2(s, row - 1, nsys)]);
      c[idx2(s, row, nsys)] *= r;
    }
  }
  for (int row = nrow - 2; row >= 0; --row) {
    for (int s = 0; s < nsys; ++s)
      d[idx2(s, row, nsys)] -= c[idx2(s, row, nsys)] * d[idx2(s, row + 1, nsys)];
  }
}

// ── 순환(주기) Thomas — Sherman–Morrison ─────────────────────────────────────
// (수식 설명은 헤더 참조.) 스크래치를 밖에서 받는 본체와, 할당해 주는 공개
// 래퍼로 나뉜다 — 플랜 경로는 플랜 소유 스크래치를 재사용한다 (GPU 조건).
namespace {

void thomas_cyclic_many_scratch(real_t* a, real_t* b, real_t* c, real_t* d,
                                const int nsys, const int nrow,
                                real_t* z,        // 보조 RHS, nsys*nrow
                                real_t* gamma,    // nsys
                                real_t* factor) { // nsys
  if (nrow == 1) {
    // 자기 자신으로 랩: (a + b + c)·x = d — 2차원 축퇴(n=1 주기 축)의 실경로.
    for (int s = 0; s < nsys; ++s) d[s] /= (a[s] + b[s] + c[s]);
    return;
  }
  if (nrow == 2) {
    // 두 행이 서로가 서로의 이웃이자 랩 — 비대각이 합쳐진 2×2 를 직접 푼다.
    for (int s = 0; s < nsys; ++s) {
      const real_t m00 = b[idx2(s, 0, nsys)];
      const real_t m01 = a[idx2(s, 0, nsys)] + c[idx2(s, 0, nsys)];
      const real_t m10 = a[idx2(s, 1, nsys)] + c[idx2(s, 1, nsys)];
      const real_t m11 = b[idx2(s, 1, nsys)];
      const real_t d0 = d[idx2(s, 0, nsys)];
      const real_t d1 = d[idx2(s, 1, nsys)];
      const real_t inv_det = 1 / (m00 * m11 - m01 * m10);
      d[idx2(s, 0, nsys)] = inv_det * (d0 * m11 - m01 * d1);
      d[idx2(s, 1, nsys)] = inv_det * (m00 * d1 - d0 * m10);
    }
    return;
  }

  const int last = nrow - 1;
  const std::size_t n_all = static_cast<std::size_t>(nsys) * nrow;
  for (long long q = 0; q < static_cast<long long>(n_all); ++q) z[q] = 0;

  for (int s = 0; s < nsys; ++s) {
    gamma[s] = -b[s];
    const real_t wrap_c = c[idx2(s, last, nsys)];  // 전진이 c[last] 를 덮기 전에 사용
    b[s] -= gamma[s];
    b[idx2(s, last, nsys)] -= a[s] * wrap_c / gamma[s];
    z[s] = gamma[s];
    z[idx2(s, last, nsys)] = wrap_c;
  }

  // 융합 두-RHS Thomas: d 와 z 가 같은 소거를 공유한다.
  // a[row 0](= 랩 계수 a₀)는 어느 패스도 건드리지 않아 마지막 보정에서 그대로 쓴다.
  for (int s = 0; s < nsys; ++s) {
    const real_t inv = 1 / b[s];
    c[s] *= inv;
    d[s] *= inv;
    z[s] *= inv;
  }
  for (int row = 1; row < nrow; ++row) {
    for (int s = 0; s < nsys; ++s) {
      const real_t r = 1 / (b[idx2(s, row, nsys)]
                            - a[idx2(s, row, nsys)] * c[idx2(s, row - 1, nsys)]);
      d[idx2(s, row, nsys)] = r * (d[idx2(s, row, nsys)]
                                   - a[idx2(s, row, nsys)] * d[idx2(s, row - 1, nsys)]);
      z[idx2(s, row, nsys)] = r * (z[idx2(s, row, nsys)]
                                   - a[idx2(s, row, nsys)] * z[idx2(s, row - 1, nsys)]);
      c[idx2(s, row, nsys)] *= r;
    }
  }
  for (int row = nrow - 2; row >= 0; --row) {
    for (int s = 0; s < nsys; ++s) {
      d[idx2(s, row, nsys)] -= c[idx2(s, row, nsys)] * d[idx2(s, row + 1, nsys)];
      z[idx2(s, row, nsys)] -= c[idx2(s, row, nsys)] * z[idx2(s, row + 1, nsys)];
    }
  }

  for (int s = 0; s < nsys; ++s) {
    const real_t v_last = a[s] / gamma[s];  // v = (1, 0, …, 0, a₀/γ)
    const real_t v_dot_y = d[s] + v_last * d[idx2(s, last, nsys)];
    const real_t v_dot_z = z[s] + v_last * z[idx2(s, last, nsys)];
    factor[s] = v_dot_y / (1 + v_dot_z);
  }
  for (int row = 0; row < nrow; ++row) {
    for (int s = 0; s < nsys; ++s)
      d[idx2(s, row, nsys)] -= factor[s] * z[idx2(s, row, nsys)];
  }
}

}  // namespace

void thomas_cyclic_many(real_t* a, real_t* b, real_t* c, real_t* d,
                        const int nsys, const int nrow) {
  std::vector<real_t> scratch(static_cast<std::size_t>(nsys) * nrow + 2 * nsys);
  thomas_cyclic_many_scratch(a, b, c, d, nsys, nrow, scratch.data(),
                             scratch.data() + static_cast<std::size_t>(nsys) * nrow,
                             scratch.data() + static_cast<std::size_t>(nsys) * nrow
                                 + nsys);
}

// ── 분할 플랜 ────────────────────────────────────────────────────────────────

void TdmaPlan::create(const int nsys_in, MPI_Comm line_comm) {
  destroy();
  if (nsys_in <= 0)
    throw std::invalid_argument("TdmaPlan::create requires nsys > 0, got "
                                + std::to_string(nsys_in));

  MPI_Comm_dup(line_comm, &comm);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  nsys = nsys_in;

  // 라인 컴의 전 랭크가 같은 nsys 인지 — 어긋나면 alltoallv 가 조용히 데이터를
  // 섞으므로 여기서 크게 죽는다. (데카르트 한 축 컴이면 자동으로 같다.)
  int nsys_min = 0;
  int nsys_max = 0;
  MPI_Allreduce(&nsys, &nsys_min, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&nsys, &nsys_max, 1, MPI_INT, MPI_MAX, comm);
  if (nsys_min != nsys_max) {
    destroy();
    throw std::invalid_argument(
        "TdmaPlan::create: every rank in the line communicator must pass the same"
        " nsys (saw " + std::to_string(nsys_min) + " .. " + std::to_string(nsys_max)
        + ")");
  }

  // 축약계 시스템 분배 — 분할 규칙은 격자 분할과 같은 split_evenly 하나만 쓴다.
  // nsys < nprocs 면 split_evenly 가 throw (빈 랭크 금지).
  chunk_first.assign(nprocs, 0);
  chunk_count.assign(nprocs, 0);
  for (int r = 0; r < nprocs; ++r) {
    const parallel::ParaRange range = parallel::split_evenly(nsys, nprocs, r);
    chunk_first[r] = range.offset;
    chunk_count[r] = range.count;
  }
  my_chunk_count = chunk_count[rank];

  // 전진 = 양끝 행 2개 × 배열 3개(a,c,d), 후진 = 해 1개 × 양끝 행 2개.
  forward_send_counts.assign(nprocs, 0);
  forward_send_displs.assign(nprocs, 0);
  forward_recv_counts.assign(nprocs, 0);
  forward_recv_displs.assign(nprocs, 0);
  backward_send_counts.assign(nprocs, 0);
  backward_send_displs.assign(nprocs, 0);
  backward_recv_counts.assign(nprocs, 0);
  backward_recv_displs.assign(nprocs, 0);
  int forward_send_total = 0;
  int forward_recv_total = 0;
  int backward_send_total = 0;
  int backward_recv_total = 0;
  for (int r = 0; r < nprocs; ++r) {
    forward_send_counts[r] = 6 * chunk_count[r];
    forward_recv_counts[r] = 6 * my_chunk_count;
    backward_send_counts[r] = 2 * my_chunk_count;
    backward_recv_counts[r] = 2 * chunk_count[r];
    forward_send_displs[r] = forward_send_total;
    forward_recv_displs[r] = forward_recv_total;
    backward_send_displs[r] = backward_send_total;
    backward_recv_displs[r] = backward_recv_total;
    forward_send_total += forward_send_counts[r];
    forward_recv_total += forward_recv_counts[r];
    backward_send_total += backward_send_counts[r];
    backward_recv_total += backward_recv_counts[r];
  }

  edge_a.assign(static_cast<std::size_t>(2) * nsys, 0.0);
  edge_c.assign(static_cast<std::size_t>(2) * nsys, 0.0);
  edge_d.assign(static_cast<std::size_t>(2) * nsys, 0.0);
  const std::size_t reduced_size =
      static_cast<std::size_t>(my_chunk_count) * 2 * nprocs;
  reduced_a.assign(reduced_size, 0.0);
  reduced_b.assign(reduced_size, 0.0);
  reduced_c.assign(reduced_size, 0.0);
  reduced_d.assign(reduced_size, 0.0);
  send_buffer.assign(static_cast<std::size_t>(
                         std::max(forward_send_total, backward_send_total)), 0.0);
  recv_buffer.assign(static_cast<std::size_t>(
                         std::max(forward_recv_total, backward_recv_total)), 0.0);

  created = true;
}

void TdmaPlan::destroy() noexcept {
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized) MPI_Finalized(&finalized);
  if (initialized && !finalized && comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
  comm = MPI_COMM_NULL;
  rank = 0;
  nprocs = 1;
  nsys = 0;
  my_chunk_count = 0;
  created = false;
}

// ── 분할 풀이 공통 몸체 ──────────────────────────────────────────────────────

namespace {

void check_solve_arguments(const char* who, const TdmaPlan& plan,
                           const int nsys, const int nrow) {
  if (!plan.created)
    throw std::runtime_error(std::string(who) + ": plan has not been created");
  if (nsys != plan.nsys)
    throw std::invalid_argument(std::string(who) + ": nsys " + std::to_string(nsys)
                                + " does not match the plan ("
                                + std::to_string(plan.nsys) + ")");
  if (nrow <= 0)
    throw std::invalid_argument(std::string(who) + ": nrow must be positive, got "
                                + std::to_string(nrow));
  if (plan.nprocs > 1 && nrow < 2)
    throw std::invalid_argument(std::string(who)
                                + ": the distributed path needs nrow >= 2 per rank"
                                  " (both edge rows become reduced-system rows)");
}

// 플랜 소유 순환 스크래치를 (aux, gamma, factor) 로 잘라 준다 (lazy 크기 잡기).
void cyclic_scratch_pointers(TdmaPlan& plan, const int nsys, const int nrow,
                             real_t*& z, real_t*& gamma, real_t*& factor) {
  const std::size_t need = static_cast<std::size_t>(nsys) * nrow + 2 * nsys;
  if (plan.cyclic_scratch.size() < need) plan.cyclic_scratch.resize(need);
  z = plan.cyclic_scratch.data();
  gamma = z + static_cast<std::size_t>(nsys) * nrow;
  factor = gamma + nsys;
}

// 1단계: 로컬 소거(수정 Thomas). 끝나면
//   내부 행(1..nrow−2):  x_j + a_j·x_first + c_j·x_last = d_j   (되살리기용)
//   양끝 행 → plan.edge_*:  row2 = 0(첫)  : a′·x_{이전랭크 끝} + x_first + c′·x_last = d′
//                           row2 = 1(끝)  : a′·x_first + x_last + c′·x_{다음랭크 첫} = d′
// 양끝 행들의 결합 상대가 축약계 순서(랭크0 첫, 랭크0 끝, 랭크1 첫, …)에서 정확히
// 이웃이라 축약계가 삼중대각이 된다. 주기면 랭크0 첫 행의 a′ 상대가 마지막
// 랭크의 끝 — 즉 축약계의 "랩"이 된다 (b′ 는 전부 1 — unpack 에서 채운다).
void reduce_to_edge_rows(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                         const int nrow) {
  const int nsys = plan.nsys;
  const int last = nrow - 1;
  real_t* edge_a = plan.edge_a.data();
  real_t* edge_c = plan.edge_c.data();
  real_t* edge_d = plan.edge_d.data();

  // 행 0·1 정규화. 행 0 의 a 는 이전 랭크로 가는 결합이라 그대로 남기고,
  // 행 1부터의 전진 소거는 a-결합을 x_first(행 0) 앵커로 만든다.
  for (int row = 0; row <= 1; ++row) {
    for (int s = 0; s < nsys; ++s) {
      const real_t inv = 1 / b[idx2(s, row, nsys)];
      a[idx2(s, row, nsys)] *= inv;
      c[idx2(s, row, nsys)] *= inv;
      d[idx2(s, row, nsys)] *= inv;
    }
  }
  for (int row = 2; row < nrow; ++row) {
    for (int s = 0; s < nsys; ++s) {
      const real_t r = 1 / (b[idx2(s, row, nsys)]
                            - a[idx2(s, row, nsys)] * c[idx2(s, row - 1, nsys)]);
      d[idx2(s, row, nsys)] = r * (d[idx2(s, row, nsys)]
                                   - a[idx2(s, row, nsys)] * d[idx2(s, row - 1, nsys)]);
      c[idx2(s, row, nsys)] *= r;
      a[idx2(s, row, nsys)] = -r * a[idx2(s, row, nsys)] * a[idx2(s, row - 1, nsys)];
    }
  }
  // 후진 소거: 내부 행의 c-결합을 x_last(행 nrow−1) 앵커로. 행 nrow−2 는 이미
  // (x_first, x_last) 결합뿐이라 그대로 두고 nrow−3 부터 내려온다.
  for (int row = nrow - 3; row >= 1; --row) {
    for (int s = 0; s < nsys; ++s) {
      const real_t c_row = c[idx2(s, row, nsys)];
      d[idx2(s, row, nsys)] -= c_row * d[idx2(s, row + 1, nsys)];
      a[idx2(s, row, nsys)] -= c_row * a[idx2(s, row + 1, nsys)];
      c[idx2(s, row, nsys)] = -c_row * c[idx2(s, row + 1, nsys)];
    }
  }
  // 행 0 마무리: 행 1 을 대입해 c-결합을 x_1 → x_last 로 바꾼다.
  // nrow == 2 면 행 1 이 곧 x_last 라 정규화 형태가 이미 축약 행이다 — 생략.
  // (참고: 같은 폴더의 CUDA 포트(pascal_tdma_cuda.cu)는 이 생략이 없어
  //  nrow == 2 를 틀리게 처리한다 — M6-GPU 숙제.)
  if (nrow >= 3) {
    for (int s = 0; s < nsys; ++s) {
      const real_t r = 1 / (1 - a[idx2(s, 1, nsys)] * c[idx2(s, 0, nsys)]);
      d[idx2(s, 0, nsys)] = r * (d[idx2(s, 0, nsys)]
                                 - c[idx2(s, 0, nsys)] * d[idx2(s, 1, nsys)]);
      a[idx2(s, 0, nsys)] *= r;
      c[idx2(s, 0, nsys)] = -r * c[idx2(s, 0, nsys)] * c[idx2(s, 1, nsys)];
    }
  }

  for (int s = 0; s < nsys; ++s) {
    edge_a[idx2(s, 0, nsys)] = a[idx2(s, 0, nsys)];
    edge_c[idx2(s, 0, nsys)] = c[idx2(s, 0, nsys)];
    edge_d[idx2(s, 0, nsys)] = d[idx2(s, 0, nsys)];
    edge_a[idx2(s, 1, nsys)] = a[idx2(s, last, nsys)];
    edge_c[idx2(s, 1, nsys)] = c[idx2(s, last, nsys)];
    edge_d[idx2(s, 1, nsys)] = d[idx2(s, last, nsys)];
  }
}

// 2~4단계: 전치 → 축약계 풀이 → 반송. 주기/비주기의 차이는 축약계를 순환
// Thomas 로 푸느냐뿐이다 (cyclic_reduced).
void solve_reduced_system(TdmaPlan& plan, const bool cyclic_reduced) {
  const int nsys = plan.nsys;
  const int my = plan.my_chunk_count;
  const int nprocs = plan.nprocs;
  real_t* send = plan.send_buffer.data();
  real_t* recv = plan.recv_buffer.data();
  real_t* edge_a = plan.edge_a.data();
  real_t* edge_c = plan.edge_c.data();
  real_t* edge_d = plan.edge_d.data();
  real_t* reduced_a = plan.reduced_a.data();
  real_t* reduced_b = plan.reduced_b.data();
  real_t* reduced_c = plan.reduced_c.data();
  real_t* reduced_d = plan.reduced_d.data();

  // 전치 와이어 포맷 (랭크 r 블록): [a 섹션 | c 섹션 | d 섹션],
  // 각 섹션은 [q + row2 * 시스템수] — 레이아웃과 같은 sys-빠름 순서.
  for (int r = 0; r < nprocs; ++r) {
    const int offset = plan.forward_send_displs[r];
    const int count = plan.chunk_count[r];
    const int first = plan.chunk_first[r];
    for (int row2 = 0; row2 <= 1; ++row2) {
      for (int q = 0; q < count; ++q) {
        const std::size_t source = idx2(first + q, row2, nsys);
        send[offset + 0 * 2 * count + q + row2 * count] = edge_a[source];
        send[offset + 1 * 2 * count + q + row2 * count] = edge_c[source];
        send[offset + 2 * 2 * count + q + row2 * count] = edge_d[source];
      }
    }
  }
  MPI_Alltoallv(plan.send_buffer.data(), plan.forward_send_counts.data(),
                plan.forward_send_displs.data(), MPI_DOUBLE,
                plan.recv_buffer.data(), plan.forward_recv_counts.data(),
                plan.forward_recv_displs.data(), MPI_DOUBLE, plan.comm);

  // 축약계 조립: 행 번호 = 2·(보낸 랭크) + row2 — "랭크0 첫, 랭크0 끝, …" 순서.
  const std::size_t reduced_size = plan.reduced_b.size();
  for (long long q = 0; q < static_cast<long long>(reduced_size); ++q)
    reduced_b[q] = 1.0;
  for (int r = 0; r < nprocs; ++r) {
    const int offset = plan.forward_recv_displs[r];
    for (int row2 = 0; row2 <= 1; ++row2) {
      for (int q = 0; q < my; ++q) {
        const std::size_t target = idx2(q, 2 * r + row2, my);
        reduced_a[target] = recv[offset + 0 * 2 * my + q + row2 * my];
        reduced_c[target] = recv[offset + 1 * 2 * my + q + row2 * my];
        reduced_d[target] = recv[offset + 2 * 2 * my + q + row2 * my];
      }
    }
  }

  if (cyclic_reduced) {
    real_t *z, *gamma, *factor;
    cyclic_scratch_pointers(plan, my, 2 * nprocs, z, gamma, factor);
    thomas_cyclic_many_scratch(reduced_a, reduced_b, reduced_c, reduced_d, my,
                               2 * nprocs, z, gamma, factor);
  } else {
    thomas_many(reduced_a, reduced_b, reduced_c, reduced_d, my, 2 * nprocs);
  }

  // 해 반송: 랭크 r 에게는 그 랭크의 두 행(2r, 2r+1)만 보낸다.
  for (int r = 0; r < nprocs; ++r) {
    const int offset = plan.backward_send_displs[r];
    for (int row2 = 0; row2 <= 1; ++row2)
      for (int q = 0; q < my; ++q)
        send[offset + q + row2 * my] = reduced_d[idx2(q, 2 * r + row2, my)];
  }
  MPI_Alltoallv(plan.send_buffer.data(), plan.backward_send_counts.data(),
                plan.backward_send_displs.data(), MPI_DOUBLE,
                plan.recv_buffer.data(), plan.backward_recv_counts.data(),
                plan.backward_recv_displs.data(), MPI_DOUBLE, plan.comm);
  for (int r = 0; r < nprocs; ++r) {
    const int offset = plan.backward_recv_displs[r];
    const int count = plan.chunk_count[r];
    const int first = plan.chunk_first[r];
    for (int row2 = 0; row2 <= 1; ++row2)
      for (int q = 0; q < count; ++q)
        edge_d[idx2(first + q, row2, nsys)] = recv[offset + q + row2 * count];
  }
}

// 5단계: 양끝 해를 쓰고 내부 행을 되살린다.
void apply_edge_solutions(TdmaPlan& plan, const real_t* a, const real_t* c,
                          real_t* d, const int nrow) {
  const int nsys = plan.nsys;
  const real_t* edge_d = plan.edge_d.data();
  for (int s = 0; s < nsys; ++s) {
    d[idx2(s, 0, nsys)] = edge_d[idx2(s, 0, nsys)];
    d[idx2(s, nrow - 1, nsys)] = edge_d[idx2(s, 1, nsys)];
  }
  for (int row = 1; row < nrow - 1; ++row)
    for (int s = 0; s < nsys; ++s)
      d[idx2(s, row, nsys)] -= a[idx2(s, row, nsys)] * edge_d[idx2(s, 0, nsys)]
                               + c[idx2(s, row, nsys)] * edge_d[idx2(s, 1, nsys)];
}

}  // namespace

void solve_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                const int nsys, const int nrow) {
  check_solve_arguments("solve_many", plan, nsys, nrow);
  if (plan.nprocs == 1) {
    thomas_many(a, b, c, d, nsys, nrow);
    return;
  }
  reduce_to_edge_rows(plan, a, b, c, d, nrow);
  solve_reduced_system(plan, /*cyclic_reduced=*/false);
  apply_edge_solutions(plan, a, c, d, nrow);
}

void solve_cyclic_many(TdmaPlan& plan, real_t* a, real_t* b, real_t* c, real_t* d,
                       const int nsys, const int nrow) {
  check_solve_arguments("solve_cyclic_many", plan, nsys, nrow);
  if (plan.nprocs == 1) {
    real_t *z, *gamma, *factor;
    cyclic_scratch_pointers(plan, nsys, nrow, z, gamma, factor);
    thomas_cyclic_many_scratch(a, b, c, d, nsys, nrow, z, gamma, factor);
    return;
  }
  // 로컬 소거·되살리기는 비주기와 동일 — 랩 계수는 양끝 행의 a′/c′ 에 실려
  // 축약계로 넘어가고, 그래서 축약계만 순환 Thomas 로 바뀐다 (헤더 §2).
  reduce_to_edge_rows(plan, a, b, c, d, nrow);
  solve_reduced_system(plan, /*cyclic_reduced=*/true);
  apply_edge_solutions(plan, a, c, d, nrow);
}

}  // namespace mpmstd::pascal_tdma
