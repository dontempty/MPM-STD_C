#pragma once
// 압력 Poisson 의 x·y 분할(펜슬) 지원 — 같은 z 블록을 가진 랭크들(xy-평면 그룹, P = npx·npy)
// 안에서 alltoallv 전치로 x 라인·y 라인을 완결시킨다 (2026-10-20).
//
//   블록 (lx, ly, lz)  ──①──▶  x-펜슬: x 전부 × y 를 P 등분한 조각 × lz   → x FFT
//   x-펜슬 스펙트럼     ──②──▶  y-펜슬: x 모드를 P 등분한 조각 × y 전부 × lz → y FFT → z TDMA
//   (역순으로 ②⁻¹ → 역 x FFT → ①⁻¹)
//
// z 는 처음부터 끝까지 원래 z 분할 그대로 — z 삼중대각은 기존 분산 TDMA(z 통신자)가 푼다.
// 같은 xy 좌표의 z 랭크들은 같은 x 모드 조각을 가지므로 TDMA 시스템 집합이 일치한다.
// 전치는 호스트 버퍼에서 한다 (GPU 빌드는 장·스펙트럼을 내려받아 전치 후 다시 올린다 —
// 정확성 우선; npx = npy = 1 이면 이 경로를 아예 안 쓴다).
//
// 레이아웃 (solve.hpp 의 슬랩 레이아웃과 같고, y·모드 개수만 로컬 조각):
//   x 라인   : [(jl·nz + k)·line + i]          jl ∈ y 조각, line = Nx (주기) / 2Nx (거울)
//   x 스펙트럼: 2 × [(jl·nz + k)·Nxm + mx]     Nxm = 전체 x 모드 수
//   y 스펙트럼: 2 × [(k·nml + mxl)·Ny + jy]    nml = 로컬 x 모드 수, mxl = mx − 모드 오프셋
#include <mpi.h>

#include <vector>

#include "common/types.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::pressure {

struct PencilLayout {
  MPI_Comm comm = MPI_COMM_NULL;  // xy-평면 그룹 (같은 z 블록)
  int size = 1, rank = 0;
  int n_x = 0, n_y = 0, nz_local = 0, n_mode_x = 0;
  // 그룹 랭크 r 의 블록 범위 (전역 내부 번호)
  std::vector<int> block_x_offset, block_x_count, block_y_offset, block_y_count;
  // x-펜슬 y 조각, y-펜슬 x 모드 조각
  std::vector<int> y_offset, y_count, mode_offset, mode_count;

  PencilLayout() = default;
  PencilLayout(const parallel::Subdomain& sub, int n_mode_x);
  ~PencilLayout();
  PencilLayout(const PencilLayout&) = delete;
  PencilLayout& operator=(const PencilLayout&) = delete;

  int my_y_offset() const { return y_offset[rank]; }
  int my_y_count() const { return y_count[rank]; }
  int my_mode_offset() const { return mode_offset[rank]; }
  int my_mode_count() const { return mode_count[rank]; }
};

// ① 장 내부(블록) → x 라인 (길이 line_length; 거울이면 앞 Nx 에 값, 뒤 Nx 에 거울)
//    field_host 는 Field::download 한 전체 배열 (ghost 포함, (i·nyt + j)·nzt + k).
void block_to_x_lines(const PencilLayout& layout, const parallel::Subdomain& sub,
                      const std::vector<real_t>& field_host, bool mirrored,
                      std::vector<real_t>& lines);
// ①⁻¹ x 라인의 앞 Nx 개 × scale − subtract → 장 내부(블록). ghost 는 건드리지 않는다.
void x_lines_to_block(const PencilLayout& layout, const parallel::Subdomain& sub,
                      const std::vector<real_t>& lines, int line_length, real_t scale,
                      real_t subtract, std::vector<real_t>& field_host);
// ② x 스펙트럼(y 조각) → y 스펙트럼(x 모드 조각), ②⁻¹ 은 역방향
void x_spectrum_to_y_spectrum(const PencilLayout& layout, const std::vector<real_t>& sx,
                              std::vector<real_t>& sxy);
void y_spectrum_to_x_spectrum(const PencilLayout& layout, const std::vector<real_t>& sxy,
                              std::vector<real_t>& sx);

}  // namespace mpmstd::equation::pressure
