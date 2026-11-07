#pragma once
// 연속 배치 1차원 FFT — cuFFT(GPU) / FFTW(CPU). 비정규화 (정·역 왕복 = length 배). 복소 = double 쌍.
//   BatchedComplexFft : 제자리 c2c, 원소 [b·length + i]. length = 1 이면 항등.
//   BatchedRealFft    : lines [b·length + x] (실수) ↔ spectrum [b·(length/2+1) + mx] (복소). 역변환은 spectrum 을 파괴한다.
// FFTW 플랜은 생성 때 준 버퍼에 묶인다. Buffer 는 재할당하지 않으므로 그대로 쓴다.

#ifdef MPMSTD_GPU_FFT
#include <cufft.h>
#else
#include <fftw3.h>
#endif

#include "common/types.hpp"

namespace fft_test {

using mpmstd::real_t;

class BatchedComplexFft {
 public:
  BatchedComplexFft(int length, long long batch, real_t* buffer);
  ~BatchedComplexFft();
  BatchedComplexFft(const BatchedComplexFft&) = delete;
  BatchedComplexFft& operator=(const BatchedComplexFft&) = delete;

  void forward();   // 지수 부호 −
  void backward();  // 지수 부호 +

 private:
  real_t* buffer_ = nullptr;
  bool trivial_ = true;
#ifdef MPMSTD_GPU_FFT
  cufftHandle plan_ = 0;
#else
  fftw_plan forward_ = nullptr, backward_ = nullptr;
#endif
};

class BatchedRealFft {
 public:
  BatchedRealFft(int length, long long batch, real_t* lines, real_t* spectrum);
  ~BatchedRealFft();
  BatchedRealFft(const BatchedRealFft&) = delete;
  BatchedRealFft& operator=(const BatchedRealFft&) = delete;

  void forward();   // lines → spectrum
  void backward();  // spectrum → lines

 private:
  real_t* lines_ = nullptr;
  real_t* spectrum_ = nullptr;
#ifdef MPMSTD_GPU_FFT
  cufftHandle plan_forward_ = 0, plan_backward_ = 0;
#else
  fftw_plan forward_ = nullptr, backward_ = nullptr;
#endif
};

}  // namespace fft_test
