#pragma once
// 필드의 POD 뷰 — 포인터 + 치수의 값 복사라서 CUDA 커널 인자로 그대로 들어간다
// (Field 객체 자체는 호스트 스택에 살아 커널에서 못 만진다). 핫루프/커널 본문이
// Field 와 같은 (i, j, k) 문법을 유지하게 해 주는 유일한 장치다.
// CPU 뷰와 GPU 뷰는 같은 타입이다 — 포인터가 호스트냐 디바이스냐만 다르다.

#include <cstddef>

#include "common/hd.hpp"
#include "common/types.hpp"

namespace mpmstd::field {

struct View {
  real_t* p = nullptr;
  int ny = 0;
  int nz = 0;
  MPMSTD_HD real_t& operator()(int i, int j, int k) const {
    return p[(static_cast<std::size_t>(i) * ny + j) * nz + k];
  }
};

struct ConstView {
  const real_t* p = nullptr;
  int ny = 0;
  int nz = 0;
  ConstView() = default;
  MPMSTD_HD ConstView(const real_t* data, int ny_total, int nz_total)
      : p(data), ny(ny_total), nz(nz_total) {}
  MPMSTD_HD ConstView(const View& v) : p(v.p), ny(v.ny), nz(v.nz) {}  // 쓰기 → 읽기
  MPMSTD_HD real_t operator()(int i, int j, int k) const {
    return p[(static_cast<std::size_t>(i) * ny + j) * nz + k];
  }
};

}  // namespace mpmstd::field
