#pragma once
// post/spectra 의 백엔드 커널 (내부 전용) — CPU(spectra_kernels.cpp)·GPU(spectra_kernels.cu) 쌍둥이.
#include <cstddef>

#include "common/types.hpp"

namespace mpmstd::post {

// power[line][m] += scale·c_m·|X[line][m]|², c_0 = c_{n/2}(n 짝수) = 1, 그 외 2 (한쪽 스펙트럼).
// spectrum 은 (실,허) 쌍 평탄 배열 [line][m], n_lines 개 라인, 라인당 n_mode 모드, 원 라인 길이 n.
void accumulate_line_power(real_t* power, const real_t* spectrum, int n_lines, int n_mode, int n,
                           real_t scale);
void zero_buffer(real_t* data, std::size_t n);
// staging[line·length + i] = in[i·stride + line] — 필드 메모리의 strided 라인 n_lines 개(배치 = 인접 k)를
// 연속 배치로 모은다 (FFT 는 연속 레이아웃만 쓴다: cuFFT 가 "원소 보폭 > 배치 간격" 레이아웃을 실행 시 거부, 2026-10-09).
void gather_lines(real_t* staging, const real_t* in, int length, int n_lines, long long stride);

}  // namespace mpmstd::post
