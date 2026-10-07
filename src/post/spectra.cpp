#include "post/spectra.hpp"

#include <mpi.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef MPMSTD_GPU_FFT
#include <cuda_runtime.h>
#endif

#include "post/spectra_kernels.hpp"

// 오케스트레이션 전용 TU — 양쪽 빌드 공용 (pressure/solve.cpp 와 같은 규칙: FFT 실행 래퍼만 ifdef,
// 파워 누적은 spectra_kernels 쌍둥이). 라인은 필드 메모리를 strided 로 직접 읽는다:
//   x 라인(고정 j): 원소 (i,j,k) = (i·ny_t + j)·nz_t + k → i 보폭 ny_t·nz_t, 배치 k (보폭 1), 길이 nx
//   y 라인(고정 i): j 보폭 nz_t, 배치 k (보폭 1), 길이 ny
// 한 호출 = 고정 j(또는 i) 의 nz_local 라인 배치 → j(또는 i) 루프로 ny(nx) 번 실행.

namespace mpmstd::post {

namespace {

#ifdef MPMSTD_GPU_FFT
void check_cufft(const cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS)
    throw std::runtime_error(std::string("cuFFT failed (spectra): ") + what);
}
#endif

}  // namespace

LineSpectrumPlan::LineSpectrumPlan(const parallel::Subdomain& sub) {
  if (sub.topology().x().nprocs != 1 || sub.topology().y().nprocs != 1)
    throw std::invalid_argument("LineSpectrumPlan: npx = npy = 1 required (lines must be local)");
  nx_ = sub.x().n_local;
  ny_ = sub.y().n_local;
  nz_local_ = sub.z().n_local;
  ny_total_ = sub.y().n_total;
  nz_total_ = sub.z().n_total;
  n_mode_x_ = nx_ / 2 + 1;
  n_mode_y_ = ny_ / 2 + 1;
  const std::size_t n_complex = static_cast<std::size_t>(nz_local_) * std::max(n_mode_x_, n_mode_y_);
  spectrum_.assign(2 * n_complex);
  power_x_.assign(static_cast<std::size_t>(nz_local_) * n_mode_x_);
  power_y_.assign(static_cast<std::size_t>(nz_local_) * n_mode_y_);
  int length_x = nx_, length_y = ny_;
  const int stride_x = ny_total_ * nz_total_, stride_y = nz_total_;
#ifdef MPMSTD_GPU_FFT
  int embed_x = nx_, embed_y = ny_, embed_mx = n_mode_x_, embed_my = n_mode_y_;
  check_cufft(cufftPlanMany(&plan_x_, 1, &length_x, &embed_x, stride_x, 1, &embed_mx, 1,
                            n_mode_x_, CUFFT_D2Z, nz_local_),
              "PlanMany D2Z x-lines");
  check_cufft(cufftPlanMany(&plan_y_, 1, &length_y, &embed_y, stride_y, 1, &embed_my, 1,
                            n_mode_y_, CUFFT_D2Z, nz_local_),
              "PlanMany D2Z y-lines");
#else
  // 플랜은 대표 주소로 만들고 실행은 fftw_execute_dft_r2c 로 라인마다 다른 주소에 — 정렬 가정을
  // 끄기 위해 FFTW_UNALIGNED. 입력은 보존된다(out-of-place r2c 기본).
  std::vector<real_t> probe_in(static_cast<std::size_t>(stride_x) * nx_ + nz_local_, 0);
  probe_in.resize(std::max(probe_in.size(), static_cast<std::size_t>(stride_y) * ny_ + nz_local_));
  plan_x_ = fftw_plan_many_dft_r2c(1, &length_x, nz_local_, probe_in.data(), nullptr, stride_x, 1,
                                   reinterpret_cast<fftw_complex*>(spectrum_.data()), nullptr, 1,
                                   n_mode_x_, FFTW_ESTIMATE | FFTW_UNALIGNED);
  plan_y_ = fftw_plan_many_dft_r2c(1, &length_y, nz_local_, probe_in.data(), nullptr, stride_y, 1,
                                   reinterpret_cast<fftw_complex*>(spectrum_.data()), nullptr, 1,
                                   n_mode_y_, FFTW_ESTIMATE | FFTW_UNALIGNED);
  if (!plan_x_ || !plan_y_) throw std::runtime_error("LineSpectrumPlan: FFTW plan failed");
#endif
}

LineSpectrumPlan::~LineSpectrumPlan() {
#ifdef MPMSTD_GPU_FFT
  if (plan_x_) cufftDestroy(plan_x_);
  if (plan_y_) cufftDestroy(plan_y_);
#else
  if (plan_x_) fftw_destroy_plan(plan_x_);
  if (plan_y_) fftw_destroy_plan(plan_y_);
#endif
}

void LineSpectrumPlan::compute(const field::Field& F, const parallel::Subdomain& sub,
                               LineSpectra& out) {
  const int h = halo_width;
  zero_buffer(power_x_.data(), power_x_.size());
  zero_buffer(power_y_.data(), power_y_.size());
  real_t* base = const_cast<real_t*>(F.data());  // FFT 라이브러리 시그니처 때문 — 입력은 보존된다
  const real_t scale_x = real_t(1) / (static_cast<real_t>(nx_) * nx_ * ny_);  // 1/n² × 1/라인수
  const real_t scale_y = real_t(1) / (static_cast<real_t>(ny_) * ny_ * nx_);
  // x 라인: 고정 j, 배치 k = h..h+nz_local−1, 시작 원소 (h, j, h)
  for (int j = h; j < h + ny_; ++j) {
    real_t* in = base + (static_cast<std::size_t>(h) * ny_total_ + j) * nz_total_ + h;
#ifdef MPMSTD_GPU_FFT
    check_cufft(cufftExecD2Z(plan_x_, in, reinterpret_cast<cufftDoubleComplex*>(spectrum_.data())),
                "ExecD2Z x");
    cudaDeviceSynchronize();
#else
    fftw_execute_dft_r2c(plan_x_, in, reinterpret_cast<fftw_complex*>(spectrum_.data()));
#endif
    accumulate_line_power(power_x_.data(), spectrum_.data(), nz_local_, n_mode_x_, nx_, scale_x);
  }
  // y 라인: 고정 i, 배치 k, 시작 원소 (i, h, h)
  for (int i = h; i < h + nx_; ++i) {
    real_t* in = base + (static_cast<std::size_t>(i) * ny_total_ + h) * nz_total_ + h;
#ifdef MPMSTD_GPU_FFT
    check_cufft(cufftExecD2Z(plan_y_, in, reinterpret_cast<cufftDoubleComplex*>(spectrum_.data())),
                "ExecD2Z y");
    cudaDeviceSynchronize();
#else
    fftw_execute_dft_r2c(plan_y_, in, reinterpret_cast<fftw_complex*>(spectrum_.data()));
#endif
    accumulate_line_power(power_y_.data(), spectrum_.data(), nz_local_, n_mode_y_, ny_, scale_y);
  }
  // 로컬 z 슬랩 → 전역 배열 (다른 랭크 몫은 0) → Allreduce
  std::vector<real_t> local_x, local_y;
  power_x_.download(local_x);
  power_y_.download(local_y);
  const int nz_global = sub.z().n_global, z_offset = sub.z().offset;
  out.nz_global = nz_global;
  out.n_mode_x = n_mode_x_;
  out.n_mode_y = n_mode_y_;
  out.x.assign(static_cast<std::size_t>(nz_global) * n_mode_x_, 0);
  out.y.assign(static_cast<std::size_t>(nz_global) * n_mode_y_, 0);
  for (int k = 0; k < nz_local_; ++k) {
    std::copy(local_x.begin() + static_cast<std::ptrdiff_t>(k) * n_mode_x_,
              local_x.begin() + static_cast<std::ptrdiff_t>(k + 1) * n_mode_x_,
              out.x.begin() + static_cast<std::ptrdiff_t>(z_offset + k) * n_mode_x_);
    std::copy(local_y.begin() + static_cast<std::ptrdiff_t>(k) * n_mode_y_,
              local_y.begin() + static_cast<std::ptrdiff_t>(k + 1) * n_mode_y_,
              out.y.begin() + static_cast<std::ptrdiff_t>(z_offset + k) * n_mode_y_);
  }
  MPI_Allreduce(MPI_IN_PLACE, out.x.data(), static_cast<int>(out.x.size()), MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
  MPI_Allreduce(MPI_IN_PLACE, out.y.data(), static_cast<int>(out.y.size()), MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
}

}  // namespace mpmstd::post
