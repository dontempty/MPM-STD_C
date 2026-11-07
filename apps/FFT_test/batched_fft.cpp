#include "batched_fft.hpp"

#include <climits>
#include <stdexcept>
#include <string>

namespace fft_test {

namespace {

#ifdef MPMSTD_GPU_FFT
void check_cufft(const cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS)
    throw std::runtime_error(std::string("cuFFT failed (FFT_test): ") + what + " (cufftResult "
                             + std::to_string(static_cast<int>(status)) + ")");
}
#endif

int checked_batch(const int length, const long long batch) {
  if (length <= 0 || batch <= 0) throw std::invalid_argument("FFT_test FFT: sizes must be > 0");
  if (batch > INT_MAX) throw std::invalid_argument("FFT_test FFT: batch exceeds int");
  return static_cast<int>(batch);
}

}  // namespace

BatchedComplexFft::BatchedComplexFft(const int length, const long long batch, real_t* buffer) : buffer_(buffer) {
  const int howmany = checked_batch(length, batch);
  if (length == 1) return;
  trivial_ = false;
  int n = length;
#ifdef MPMSTD_GPU_FFT
  check_cufft(cufftPlanMany(&plan_, 1, &n, nullptr, 1, n, nullptr, 1, n, CUFFT_Z2Z, howmany), "PlanMany Z2Z");
#else
  fftw_complex* data = reinterpret_cast<fftw_complex*>(buffer_);
  forward_ = fftw_plan_many_dft(1, &n, howmany, data, nullptr, 1, n, data, nullptr, 1, n, FFTW_FORWARD, FFTW_ESTIMATE);
  backward_ = fftw_plan_many_dft(1, &n, howmany, data, nullptr, 1, n, data, nullptr, 1, n, FFTW_BACKWARD, FFTW_ESTIMATE);
  if (!forward_ || !backward_) throw std::runtime_error("BatchedComplexFft: FFTW plan failed");
#endif
}

BatchedComplexFft::~BatchedComplexFft() {
#ifdef MPMSTD_GPU_FFT
  if (plan_) cufftDestroy(plan_);
#else
  if (forward_) fftw_destroy_plan(forward_);
  if (backward_) fftw_destroy_plan(backward_);
#endif
}

void BatchedComplexFft::forward() {
  if (trivial_) return;
#ifdef MPMSTD_GPU_FFT
  cufftDoubleComplex* data = reinterpret_cast<cufftDoubleComplex*>(buffer_);
  check_cufft(cufftExecZ2Z(plan_, data, data, CUFFT_FORWARD), "ExecZ2Z forward");
#else
  fftw_execute(forward_);
#endif
}

void BatchedComplexFft::backward() {
  if (trivial_) return;
#ifdef MPMSTD_GPU_FFT
  cufftDoubleComplex* data = reinterpret_cast<cufftDoubleComplex*>(buffer_);
  check_cufft(cufftExecZ2Z(plan_, data, data, CUFFT_INVERSE), "ExecZ2Z inverse");
#else
  fftw_execute(backward_);
#endif
}

BatchedRealFft::BatchedRealFft(const int length, const long long batch, real_t* lines, real_t* spectrum)
    : lines_(lines), spectrum_(spectrum) {
  const int howmany = checked_batch(length, batch);
  int n = length;
  const int n_mode = length / 2 + 1;
#ifdef MPMSTD_GPU_FFT
  check_cufft(cufftPlanMany(&plan_forward_, 1, &n, nullptr, 1, n, nullptr, 1, n_mode, CUFFT_D2Z, howmany), "PlanMany D2Z");
  check_cufft(cufftPlanMany(&plan_backward_, 1, &n, nullptr, 1, n_mode, nullptr, 1, n, CUFFT_Z2D, howmany), "PlanMany Z2D");
#else
  fftw_complex* s = reinterpret_cast<fftw_complex*>(spectrum_);
  forward_ = fftw_plan_many_dft_r2c(1, &n, howmany, lines_, nullptr, 1, n, s, nullptr, 1, n_mode, FFTW_ESTIMATE);
  backward_ = fftw_plan_many_dft_c2r(1, &n, howmany, s, nullptr, 1, n_mode, lines_, nullptr, 1, n, FFTW_ESTIMATE);
  if (!forward_ || !backward_) throw std::runtime_error("BatchedRealFft: FFTW plan failed");
#endif
}

BatchedRealFft::~BatchedRealFft() {
#ifdef MPMSTD_GPU_FFT
  if (plan_forward_) cufftDestroy(plan_forward_);
  if (plan_backward_) cufftDestroy(plan_backward_);
#else
  if (forward_) fftw_destroy_plan(forward_);
  if (backward_) fftw_destroy_plan(backward_);
#endif
}

void BatchedRealFft::forward() {
#ifdef MPMSTD_GPU_FFT
  check_cufft(cufftExecD2Z(plan_forward_, lines_, reinterpret_cast<cufftDoubleComplex*>(spectrum_)), "ExecD2Z");
#else
  fftw_execute(forward_);
#endif
}

void BatchedRealFft::backward() {
#ifdef MPMSTD_GPU_FFT
  check_cufft(cufftExecZ2D(plan_backward_, reinterpret_cast<cufftDoubleComplex*>(spectrum_), lines_), "ExecZ2D");
#else
  fftw_execute(backward_);
#endif
}

}  // namespace fft_test
