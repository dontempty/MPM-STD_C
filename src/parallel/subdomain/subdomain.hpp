#pragma once
// 전역 격자의 이 랭크 몫(분할) + halo 교환.
//
// 분할 규칙 split_evenly 는 PaScaL_TDMA_C 의 para_range 와 동일하다 — TDMA
// plan 의 extent 와 정렬되므로 **이 규칙을 바꾸면 TDMA 가 깨진다**.
// 앞쪽 랭크가 나머지를 먼저 받는다 (예: 9 셀 / 2 랭크 → 5 + 4).
//
// halo 교환은 면(face)만 한다 — 모서리(edge/corner)는 보증하지 않는다.
// 모든 스텐실이 ±1 면만 읽는 동안은 충분하다 (교차미분 확장 시 여기부터).

#include <mpi.h>

#include <cstddef>
#include <string>

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "parallel/topology/topology.hpp"

namespace mpmstd::parallel {

// 한 축의 균등 분할 결과.
struct ParaRange {
  int offset = 0;  // 첫 로컬 셀의 전역 번호 (0-based)
  int count = 0;   // 로컬 셀 수
};

// n_global 셀을 nprocs 개 블록으로 나눌 때 rank 번째 블록.
// 모든 랭크가 최소 1 셀을 받아야 한다 — 아니면 throw (MPM-STD_C 는 빈 랭크가
// 조용히 생겼다). nprocs/rank 가 말이 안 되면 throw.
ParaRange split_evenly(int n_global, int nprocs, int rank);

class Subdomain {
 public:
  // 한 축의 분할 수치 묶음 — sub.z().n_local 처럼 이름으로 집는다.
  struct AxisSize {
    int n_global = 0;  // 축 전역 내부 셀 수
    int offset = 0;    // 첫 로컬 내부 셀의 전역 번호
    int n_local = 0;   // 로컬 내부 셀 수
    int n_total = 0;   // = n_local + 2*halo_width (배열 치수)

    // 전역 ↔ 로컬 변환. offset 은 "내부 첫 셀"을 가리키고 로컬 내부는 인덱스
    // halo_width 부터 시작하므로 halo_width 보정이 들어간다 — 이 보정을
    // 손으로 쓰다 틀리는 것이 MPM-STD_C 의 단골 off-by-one 이었다.
    int global(int local_i) const { return offset + (local_i - halo_width); }
    int local(int global_i) const { return global_i - offset + halo_width; }
    // 전역 셀 global_i 가 이 랭크의 내부인가 (예: 전역 중앙 평면 소유 판정).
    bool owns(int global_i) const {
      return global_i >= offset && global_i < offset + n_local;
    }
  };

  // topo 는 Subdomain 보다 오래 살아야 한다 (main 에서 생성 순서로 보장).
  Subdomain(const Topology& topo, int nx_global, int ny_global, int nz_global);
  ~Subdomain();
  Subdomain(const Subdomain&) = delete;
  Subdomain& operator=(const Subdomain&) = delete;

  const AxisSize& x() const { return x_; }
  const AxisSize& y() const { return y_; }
  const AxisSize& z() const { return z_; }

  // 배열 한 개의 총 원소 수 = x.n_total * y.n_total * z.n_total
  std::size_t n_elements() const;

  // data: n_total 모양 배열, 선형 인덱스 (i*ny_total + j)*nz_total + k.
  // 각 축에서 첫/끝 내부 평면을 이웃과 맞바꿔 ghost 평면을 채운다.
  // 벽(이웃 = MPI_PROC_NULL)에서는 MPI 가 자동으로 아무것도 하지 않는다.
  void exchange_halo(real_t* data) const;

  const Topology& topology() const { return topo_; }

 private:
  const Topology& topo_;
  AxisSize x_;
  AxisSize y_;
  AxisSize z_;

  // x-평면(i 고정)은 메모리 연속이라 datatype 이 필요 없다.
  MPI_Datatype y_plane_type_ = MPI_DATATYPE_NULL;  // j 고정 한 장 — strided
  MPI_Datatype z_plane_type_ = MPI_DATATYPE_NULL;  // k 고정 한 장 — 가장 벌어짐

#ifdef MPMSTD_GPU
  // GPU 교환: 벌어진 y/z 평면은 "팩 커널 → 연속 디바이스 버퍼 → MPI" 로 보낸다
  // (디바이스 포인터에 subarray datatype 을 물리면 UCX 가 죽는 것을 실측 —
  //  2026-10-04; x 평면은 연속이라 필드에서 바로 보낸다). 구현은
  // subdomain_halo.cu, 버퍼는 스크래치라 mutable.
  void exchange_halo_device(real_t* f) const;
  mutable field::Buffer y_send_minus_, y_send_plus_, y_recv_minus_, y_recv_plus_;
  mutable field::Buffer z_send_minus_, z_send_plus_, z_recv_minus_, z_recv_plus_;
#endif
};

}  // namespace mpmstd::parallel
