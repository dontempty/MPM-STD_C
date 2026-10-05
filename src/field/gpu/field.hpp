#pragma once
// 3차원 배열 하나 — GPU 백엔드. 레이아웃·staggering 해석은 CPU 판
// (field/cpu/field.hpp)과 완전히 같다 (conventions.md §1·§3이 정본).
//
// 다른 점은 하나: 저장이 **디바이스 상주(cudaMalloc)** 라는 것. 호스트에서
// 원소를 읽고 쓰는 operator() 가 없으므로, 호스트 코드가 실수로 원소에 손대면
// 컴파일 에러가 난다. 계산은 전부 커널(각 모듈의 .cu 쌍둥이), 호스트와는
// upload()/download() 로만 오간다 — IC 넣을 때 한 번, 통계 출력 때 조금.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/types.hpp"
#include "field/view.hpp"

namespace mpmstd::field {

class Field {
 public:
  // ghost 포함 치수. 0 으로 초기화된다 (CPU 판과 같은 계약).
  Field(int nx_total, int ny_total, int nz_total, std::string name)
      : nx_(nx_total), ny_(ny_total), nz_(nz_total), name_(std::move(name)) {
    const int minimum = 1 + 2 * halo_width;
    if (nx_ < minimum || ny_ < minimum || nz_ < minimum)
      throw std::invalid_argument(
          "Field '" + name_ + "': each extent must be >= " + std::to_string(minimum)
          + ", got " + std::to_string(nx_) + "x" + std::to_string(ny_) + "x"
          + std::to_string(nz_));
    size_ = static_cast<std::size_t>(nx_) * ny_ * nz_;
    MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                 size_ * sizeof(real_t)));
    MPMSTD_CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(real_t)));
  }
  ~Field() {
    if (data_) cudaFree(data_);
  }
  Field(const Field&) = delete;
  Field& operator=(const Field&) = delete;

  int nx_total() const { return nx_; }
  int ny_total() const { return ny_; }
  int nz_total() const { return nz_; }
  const std::string& name() const { return name_; }
  std::size_t size() const { return size_; }

  // 디바이스 포인터 — 커널·CUDA-aware MPI(halo 교환)·cuFFT 가 그대로 쓴다.
  real_t* data() { return data_; }
  const real_t* data() const { return data_; }

  View view() { return {data_, ny_, nz_}; }
  ConstView view() const { return {data_, ny_, nz_}; }

  // 전체 칸을 value 로 (커널 — field.cu).
  void fill(real_t value);

  // 호스트 ↔ 디바이스 명시 복사 — IC 주입과 통계·IO 전용.
  void upload(const std::vector<real_t>& host) {
    if (host.size() != size_)
      throw std::invalid_argument("Field '" + name_ + "'.upload: size mismatch");
    MPMSTD_CUDA_CHECK(cudaMemcpy(data_, host.data(), size_ * sizeof(real_t),
                                 cudaMemcpyHostToDevice));
  }
  void download(std::vector<real_t>& host) const {
    host.resize(size_);
    MPMSTD_CUDA_CHECK(cudaMemcpy(host.data(), data_, size_ * sizeof(real_t),
                                 cudaMemcpyDeviceToHost));
  }

 private:
  int nx_;
  int ny_;
  int nz_;
  std::string name_;
  std::size_t size_ = 0;
  real_t* data_ = nullptr;
};

}  // namespace mpmstd::field
