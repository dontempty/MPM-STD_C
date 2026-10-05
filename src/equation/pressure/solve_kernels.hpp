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
                    int n_half_x);
void reorder_y_to_x(real_t* sx, const real_t* sxy, int n_y, int nz_local,
                    int n_half_x);
// D 팩 — (0,0) 모드 핀 행의 우변 0 도 여기서 (system 의 항등행과 짝).
void pack_spectrum_to_bands(real_t* D, const real_t* sxy, int nz_local, int n_half_x,
                            int n_y, bool pin_mean_row);
void unpack_bands_to_spectrum(real_t* sxy, const real_t* D, int nz_local,
                              int n_half_x, int n_y);
void scatter_lines_to_field(field::Field& phi, const real_t* lines, real_t scale,
                            const parallel::Subdomain& sub, int n_x, int n_y,
                            int nz_local);
real_t interior_sum(const field::Field& phi, const parallel::Subdomain& sub);
void subtract_constant(field::Field& phi, const parallel::Subdomain& sub,
                       real_t value);

}  // namespace mpmstd::equation::pressure
