#pragma once
// 3D Cartesian 프로세스 격자 + 축별 1D 서브커뮤니케이터.
// 축은 이름으로 집는다 — topo.x() / topo.y() / topo.z(). BC 프레임워크(2026-10-05,
// docs/Design/20261005_boundary_condition_framework.md)부터는 "축이 매개변수인"
// 코드(ghost 채움·밴드 fold·스윕 선택)를 위해 axis(0|1|2) 인덱스 접근자도 둔다.
// (parallel/ 의 하위 폴더 topology/·subdomain/ 는 정리용 구분일 뿐, namespace 는
//  전부 mpmstd::parallel 이다 — 계획 §4.1)

#include <mpi.h>

namespace mpmstd::parallel {

class MpiContext;

// 한 축을 따라 보는 1D 통신 정보.
// neighbor 가 MPI_PROC_NULL 이면 그쪽은 전역 경계(벽)다 — ghost 채움·TDMA 벽
// 처리·W 벽면 처리가 전부 이 판정 하나를 쓴다. 주기 축이면 양 끝이 서로
// 이웃으로 이어져 MPI_PROC_NULL 이 나오지 않는다.
struct AxisComm {
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = -1;                       // 이 축에서 내가 몇 번째 블록인가
  int nprocs = 0;                      // 이 축의 블록 수
  int minus_neighbor = MPI_PROC_NULL;  // − 쪽 이웃 랭크 (벽이면 MPI_PROC_NULL)
  int plus_neighbor = MPI_PROC_NULL;   // + 쪽 이웃 랭크
  bool periodic = false;               // 축 전체가 주기인가 (TDMA cyclic/벽 선택 —
                                       // 내부 랭크는 이웃만 보고는 알 수 없다)
};

class Topology {
 public:
  // npx*npy*npz 가 MPI 랭크 수와 다르면 throw.
  // reorder = 0 으로 만들어 랭크 배치가 런 사이에 재현 가능하다.
  Topology(const MpiContext& mpi, int npx, int npy, int npz,
           bool periodic_x, bool periodic_y, bool periodic_z);
  ~Topology();
  Topology(const Topology&) = delete;
  Topology& operator=(const Topology&) = delete;

  MPI_Comm cart_comm() const { return cart_comm_; }  // 전역 리덕션용 (CFL 등)
  const AxisComm& x() const { return x_; }
  const AxisComm& y() const { return y_; }
  const AxisComm& z() const { return z_; }
  const AxisComm& axis(const int a) const { return a == 0 ? x_ : (a == 1 ? y_ : z_); }

 private:
  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  AxisComm x_;
  AxisComm y_;
  AxisComm z_;
};

}  // namespace mpmstd::parallel
