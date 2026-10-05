#pragma once
// 솔버 작업 버퍼(밴드·스테이지·스펙트럼) — Field 와 같은 백엔드 규칙 하나:
//   CPU = 호스트 std::vector, GPU(MPMSTD_GPU) = cudaMalloc 디바이스 상주.
// data() 는 그 빌드의 TDMA·FFT·커널이 바로 받는 포인터다. 호스트 원소 접근은
// 제공하지 않는다 — 필요한 쪽(통계·검증)은 download() 로 명시적으로 가져간다.

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/types.hpp"

#ifdef MPMSTD_GPU
#include "common/cuda_check.hpp"
#endif

namespace mpmstd::field {

#ifdef MPMSTD_GPU

class Buffer {
 public:
  Buffer() = default;
  ~Buffer() {
    if (data_) cudaFree(data_);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // n 개를 확보하고 0 으로 — Solver 생성자가 한 번 부른다 (매 스텝 재사용).
  void assign(const std::size_t n) {
    if (data_) cudaFree(data_);
    size_ = n;
    MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                 n * sizeof(real_t)));
    MPMSTD_CUDA_CHECK(cudaMemset(data_, 0, n * sizeof(real_t)));
  }
  std::size_t size() const { return size_; }
  real_t* data() { return data_; }
  const real_t* data() const { return data_; }

  void upload(const std::vector<real_t>& host) {
    if (host.size() != size_) throw std::invalid_argument("Buffer.upload: size mismatch");
    MPMSTD_CUDA_CHECK(cudaMemcpy(data_, host.data(), size_ * sizeof(real_t),
                                 cudaMemcpyHostToDevice));
  }
  void download(std::vector<real_t>& host) const {
    host.resize(size_);
    MPMSTD_CUDA_CHECK(cudaMemcpy(host.data(), data_, size_ * sizeof(real_t),
                                 cudaMemcpyDeviceToHost));
  }

 private:
  real_t* data_ = nullptr;
  std::size_t size_ = 0;
};

#else

class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) noexcept = default;
  Buffer& operator=(Buffer&&) noexcept = default;

  void assign(const std::size_t n) { storage_.assign(n, 0); }
  std::size_t size() const { return storage_.size(); }
  real_t* data() { return storage_.data(); }
  const real_t* data() const { return storage_.data(); }

  void upload(const std::vector<real_t>& host) {
    if (host.size() != storage_.size())
      throw std::invalid_argument("Buffer.upload: size mismatch");
    storage_ = host;
  }
  void download(std::vector<real_t>& host) const { host = storage_; }

 private:
  std::vector<real_t> storage_;
};

#endif

}  // namespace mpmstd::field
