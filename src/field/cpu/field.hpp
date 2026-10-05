#pragma once
// 3차원 배열 하나 — U, V, W, P, T 가 전부 이 한 모양을 공유한다 (ghost 포함
// (nx_total, ny_total, nz_total); 계획 §5.1 의 A안).
//
// ★ staggering(값이 셀의 어디에 놓이는지)은 저장 모양이 아니라 "인덱스 해석"이다.
//   규칙은 하나뿐이다 — **각 성분은 자기 셀의 마이너스(−) 면을 소유한다**:
//
//     U(i,j,k) = 셀 (i,j,k) 의 −x 면 값   (x 위치 = grid.x.face[i])
//     V(i,j,k) = 셀 (i,j,k) 의 −y 면 값   (y 위치 = grid.y.face[j])
//     W(i,j,k) = 셀 (i,j,k) 의 −z 면 값   (z 위치 = grid.z.face[k])
//     P, T(i,j,k) = 셀 중심 값            (위치 = grid.*.center[*])
//
//   "면이 셀보다 하나 많다" 문제가 어떻게 수납되는지(+1 면의 행방, 벽면이
//   어느 슬롯에 앉는지)는 docs/conventions.md 의 그림이 정본이다.
//
// 선형 인덱스: (i*ny_total + j)*nz_total + k — k 가 메모리에서 가장 빠르다.
// ★ 루프는 항상 i → j → k 순서 (k 가 가장 안쪽). 순서를 뒤집으면 같은 일에
//   6.7 배 느리다 (256³ 실측, docs/conventions.md §1).
//
// Field 는 MPI 를 모른다 — 치수는 정수로 받는다 (subdomain 이 정해 준다).

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/types.hpp"
#include "field/view.hpp"

namespace mpmstd::field {

class Field {
 public:
  // ghost 포함 치수. 0 으로 초기화된다.
  Field(int nx_total, int ny_total, int nz_total, std::string name)
      : nx_(nx_total), ny_(ny_total), nz_(nz_total), name_(std::move(name)) {
    const int minimum = 1 + 2 * halo_width;  // 내부 최소 1 셀 + 양쪽 ghost
    if (nx_ < minimum || ny_ < minimum || nz_ < minimum)
      throw std::invalid_argument(
          "Field '" + name_ + "': each extent must be >= " + std::to_string(minimum)
          + ", got " + std::to_string(nx_) + "x" + std::to_string(ny_) + "x"
          + std::to_string(nz_));
    data_.assign(static_cast<std::size_t>(nx_) * ny_ * nz_, real_t{0});
  }

  int nx_total() const { return nx_; }
  int ny_total() const { return ny_; }
  int nz_total() const { return nz_; }
  const std::string& name() const { return name_; }
  std::size_t size() const { return data_.size(); }

  // 핫 루프용 — 범위 검사 없음 (i, j, k 는 [0, n*_total) 이어야 한다).
  real_t& operator()(int i, int j, int k) {
    return data_[(static_cast<std::size_t>(i) * ny_ + j) * nz_ + k];
  }
  real_t operator()(int i, int j, int k) const {
    return data_[(static_cast<std::size_t>(i) * ny_ + j) * nz_ + k];
  }

  // halo 교환·IO 용 평탄 접근: sub.exchange_halo(T.data()) 처럼 쓴다.
  real_t* data() { return data_.data(); }
  const real_t* data() const { return data_.data(); }

  View view() { return {data_.data(), ny_, nz_}; }
  ConstView view() const { return {data_.data(), ny_, nz_}; }

  // 전체 칸(ghost 포함)을 value 로 — IC·RHS 초기화용.
  void fill(real_t value) {
    for (real_t& v : data_) v = value;
  }

  // GPU 판과 같은 API — IC 주입·통계 경로가 양쪽 빌드에서 같은 코드가 되게 한다.
  // (CPU 에서는 그냥 복사다.)
  void upload(const std::vector<real_t>& host) {
    if (host.size() != data_.size())
      throw std::invalid_argument("Field '" + name_ + "'.upload: size mismatch");
    data_ = host;
  }
  void download(std::vector<real_t>& host) const { host = data_; }

 private:
  int nx_;
  int ny_;
  int nz_;
  std::string name_;
  std::vector<real_t> data_;
};

}  // namespace mpmstd::field
