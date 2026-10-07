#pragma once
// 1차원 에너지 스펙트럼 E(k_x; z), E(k_y; z) — Ng et al. (2017) JFM 825 fig.11 (premultiplied k·E_uu/u_τ²).
//   내부 셀의 x 라인(y·z 마다)·y 라인(x·z 마다)을 실수 FFT 해 한쪽(k ≥ 0) 파워를 라인 평균.
//   정규화: Σ_m E[m] = 라인 평균 ⟨F²⟩ (Parseval; m = 0 칸은 평균² 포함 — 변동 스펙트럼은 m ≥ 1).
//   k_x = 2π m/L_x, k_y = 2π m/L_y 는 노트북이 붙인다. premultiplied·u_τ² 정규화도 노트북.
// FFT 백엔드는 pressure::Solver 와 같은 규칙(MPMSTD_GPU_FFT → cuFFT, 아니면 FFTW), 라인은 필드 메모리를
// 그대로 strided 로 읽는다(복사 없음). 파워 누적은 spectra_kernels.{cpp,cu} 쌍둥이.
// npx = npy = 1 전제 (라인이 한 랭크에 완결). 결과는 전 랭크가 같은 전역 배열(z Allreduce).
#ifdef MPMSTD_GPU_FFT
#include <cufft.h>
#else
#include <fftw3.h>
#endif

#include <vector>

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::post {

struct LineSpectra {
  int nz_global = 0, n_mode_x = 0, n_mode_y = 0;  // n_mode = n/2 + 1
  std::vector<real_t> x;  // [nz_global][n_mode_x]  E(k_x; z) — y 라인 평균
  std::vector<real_t> y;  // [nz_global][n_mode_y]  E(k_y; z) — x 라인 평균
};

class LineSpectrumPlan {
 public:
  explicit LineSpectrumPlan(const parallel::Subdomain& sub);
  ~LineSpectrumPlan();
  LineSpectrumPlan(const LineSpectrumPlan&) = delete;
  LineSpectrumPlan& operator=(const LineSpectrumPlan&) = delete;

  // F 의 내부 셀 값을 그대로 쓴다 (면 변수면 면 위치의 스펙트럼).
  void compute(const field::Field& F, const parallel::Subdomain& sub, LineSpectra& out);

 private:
  int nx_ = 0, ny_ = 0, nz_local_ = 0, ny_total_ = 0, nz_total_ = 0;
  int n_mode_x_ = 0, n_mode_y_ = 0;
  field::Buffer spectrum_;  // 복소 작업 버퍼 2·nz_local·max(n_mode) (실,허 쌍)
  field::Buffer power_x_;   // [nz_local][n_mode_x] 파워 누적 (백엔드)
  field::Buffer power_y_;   // [nz_local][n_mode_y]
#ifdef MPMSTD_GPU_FFT
  cufftHandle plan_x_ = 0, plan_y_ = 0;
#else
  fftw_plan plan_x_ = nullptr, plan_y_ = nullptr;
#endif
};

}  // namespace mpmstd::post
