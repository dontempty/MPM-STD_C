#pragma once
// 순환(cyclic) 분배 레이아웃 v2 — x 는 분배하지 않고(실수 FFT), y 만 FFTU (Koopman & Bisseling 2023, Alg 2.2·2.3 의 p1 = 1).
//
// 같은 z 블록의 xy 그룹(P 랭크)이 y 를 순환 분배로 나눠 갖는다:
//   q = P,  내 좌표 s = 그룹 랭크
//   전역 y 는 랭크 y mod q 의 로컬 k = y div q 에 있다.  로컬 길이 m = ny / q,  패킷 c = m / q  ← 조건 q² | ny
//   x 는 랭크 안에서 완결 → r2c 로 모드 절반(nx/2+1)만 다룬다. 기존 솔버와 시스템 수가 정확히 같다.
//   (v1 은 x·y 2차원 순환 + c2c 로 모드가 2 배였다: 같아야 할 계산이 2 배가 된 원인.)
//
// 배열 (복소 = (실, 허) double 쌍):
//   lines      : [(k·nz + z)·nx + x]                  기존 슬랩 경로의 real_lines 와 같은 모양 (ny → m)
//   spectrum_x : [(k·nz + z)·nxm + mx]                x r2c 출력
//   spectrum_y : [(z·nxm + mx)·m + k]                 기존 reorder_x_to_y 출력. 로컬 y FFT 가 연속 배치로 돈다
//   send, recv : [r·B·c + b·c + t]                    r = 상대 그룹 랭크, b = z·nxm + mx, B = nz·nxm, t = k div q
//   spectrum   : [(b·c + t)·q + k']                   작은 q 점 FFT 의 연속 배치 = TDMA 레이아웃 [sys + z·nsys]
//   로컬 모드 번호 mode = (mx·c + t)·q + k' → 전역 (mx, y 모드 s + t·q + k'·m)   (논문 (2.10))
//   q = 1 이면 spectrum = spectrum_y (FFTU 단계가 전부 생략된다).

#include <mpi.h>

#include "common/hd.hpp"
#include "common/types.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace fft_test {

using mpmstd::real_t;

// 커널 인자로 값 복사되는 POD.
struct CyclicDims {
  int n_x = 0, n_y = 0;   // 전역 nx, ny
  int n_mode_x = 0;       // nx/2 + 1
  int q = 1, s = 0;       // 그룹 랭크 수, 내 좌표
  int m = 0, c = 0;       // 로컬 y 길이 ny/q, 패킷 길이 m/q
  int nz = 0;             // z 장 수 (nz_local)

  MPMSTD_HD long long line_elements() const { return static_cast<long long>(m) * nz * n_x; }
  MPMSTD_HD long long batches() const { return static_cast<long long>(nz) * n_mode_x; }
  MPMSTD_HD long long packet_elements() const { return batches() * c; }
  MPMSTD_HD long long spectrum_elements() const { return batches() * m; }
  MPMSTD_HD int modes_local() const { return n_mode_x * m; }

  MPMSTD_HD void global_mode(const int mode, int& mode_x, int& mode_y) const {
    mode_x = mode / m;
    const int rest = mode % m;
    mode_y = s + (rest / q) * q + (rest % q) * m;
  }
};

// 그룹 통신자 + 치수. q = 그룹 크기이고, q² | ny 가 아니면 throw.
class CyclicLayout {
 public:
  explicit CyclicLayout(const mpmstd::parallel::Subdomain& sub);
  ~CyclicLayout();
  CyclicLayout(const CyclicLayout&) = delete;
  CyclicLayout& operator=(const CyclicLayout&) = delete;

  const CyclicDims& dims() const { return dims_; }
  MPI_Comm group() const { return group_; }
  int group_size() const { return dims_.q; }
  int group_rank() const { return dims_.s; }

 private:
  MPI_Comm group_ = MPI_COMM_NULL;
  CyclicDims dims_;
};

}  // namespace fft_test
