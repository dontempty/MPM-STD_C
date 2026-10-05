#pragma once
// pressure::solve 의 백엔드 커널 묶음 (내부 전용) — CPU(solve_kernels.cpp)와
// GPU(solve_kernels.cu)가 같은 선언을 구현한다. 레이아웃은 solve.hpp 의 버퍼
// 주석이 정본이고, 복소 스펙트럼은 (실, 허) 쌍의 평탄 double 배열이다.

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::pressure {

void pack_rhs_to_lines(real_t* lines, const field::Field& rhs,
                       const parallel::Subdomain& sub, int n_x, int n_y,
                       int nz_local);
void reorder_x_to_y(real_t* sxy, const real_t* sx, int n_y, int nz_local,
                    int n_mode_x);
void reorder_y_to_x(real_t* sx, const real_t* sxy, int n_y, int nz_local,
                    int n_mode_x);
// D 팩 — (0,0) 모드 핀 행의 우변 0 도 여기서 (system 의 항등행과 짝).
void pack_spectrum_to_bands(real_t* D, const real_t* sxy, int nz_local, int n_mode_x,
                            int n_y, bool pin_mean_row);
void unpack_bands_to_spectrum(real_t* sxy, const real_t* D, int nz_local,
                              int n_mode_x, int n_y);
void scatter_lines_to_field(field::Field& phi, const real_t* lines, real_t scale,
                            const parallel::Subdomain& sub, int n_x, int n_y,
                            int nz_local);
// ── x Neumann(DCT) 경로 — 거울 확장 2N 실수 FFT 로 DCT-II/III 를 만든다 ─────────
// (cuFFT 에 DCT 가 없어 CPU/GPU 가 **같은 알고리즘**을 쓴다; solve.hpp 머리말)
//   forward : y = [x₀…x_{N−1}, x_{N−1}…x₀] → r2c_{2N} → Y[k] (N+1 복소)
//             X[k] = Re(Y[k]·e^{−iπk/2N}), k < N   (= FFTW REDFT10)
//   backward: Y[k] = X[k]·e^{+iπk/2N} (k < N), Y[N] = 0 → c2r_{2N} → 앞 N 개
//             (= FFTW REDFT01; 왕복 배율 2N → scale 에 1/(2N) 포함)
// 라인 수 = n_y·nz_local. 거울 라인은 길이 2N, 거울 스펙트럼은 (N+1) 복소.
void pack_rhs_to_mirrored_lines(real_t* lines, const field::Field& rhs,
                                const parallel::Subdomain& sub, int n_x, int n_y,
                                int nz_local);
// 거울 스펙트럼(N+1 복소/라인) → x 모드 버퍼 sx(N 복소/라인, 허수부 0)
void dct_twiddle_forward(real_t* sx, const real_t* mirror_spectrum, int n_lines, int n_x);
// x 모드 버퍼 sx(실수부만 사용) → 거울 스펙트럼
void dct_twiddle_backward(real_t* mirror_spectrum, const real_t* sx, int n_lines, int n_x);
void scatter_mirrored_lines_to_field(field::Field& phi, const real_t* lines, real_t scale,
                                     const parallel::Subdomain& sub, int n_x, int n_y,
                                     int nz_local);

real_t interior_sum(const field::Field& phi, const parallel::Subdomain& sub);
void subtract_constant(field::Field& phi, const parallel::Subdomain& sub,
                       real_t value);

}  // namespace mpmstd::equation::pressure
