#pragma once
// 압력 Poisson 의 x·y 분할(펜슬) 지원 — 같은 z 블록을 가진 랭크들(xy-평면 그룹, P = npx·npy)
// 안에서 alltoallv 전치로 x 라인·y 라인을 완결시킨다 (2026-10-20).
//
//   블록 (lx, ly, lz)  ──①──▶  x-펜슬: x 전부 × y 를 P 등분한 조각 × lz   → x FFT
//   x-펜슬 스펙트럼     ──②──▶  y-펜슬: x 모드를 P 등분한 조각 × y 전부 × lz → y FFT → z TDMA
//   (역순으로 ②⁻¹ → 역 x FFT → ①⁻¹)
//
// z 는 처음부터 끝까지 원래 z 분할 그대로 — z 삼중대각은 기존 분산 TDMA(z 통신자)가 푼다.
//
// ★ 통신 (2026-10-20 개정, PaScaL_TDMA 의 alltoallv_double 과 같은 구조):
//   팩(gather) 커널 → 송신 버퍼 → MPI_Alltoallv → 수신 버퍼 → 언팩(scatter) 커널.
//   버퍼는 field::Buffer(GPU 빌드 = 디바이스 상주)이고 **GPU 는 디바이스 포인터를 그대로 MPI 에 넘긴다**
//   (CUDA-aware MPI, GPU↔GPU). 환경변수 PASCAL_TDMA_MPI_MODE=host 일 때만 호스트 경유(디버그·대조용) —
//   PaScaL_TDMA 와 같은 스위치. 팩·언팩 순서는 생성 시 한 번 만든 **인덱스 표**(디바이스 int 배열)로
//   정하므로 불균등 분할도 커널 하나로 처리한다. 백엔드: pencil_exchange.{cpp,cu} 쌍둥이.
//
// 레이아웃 (solve.hpp 의 슬랩 레이아웃과 같고, y·모드 개수만 로컬 조각):
//   장(블록)  : Field 전체 배열 (i·nyt + j)·nzt + k (ghost 포함; 내부만 읽고 쓴다)
//   x 라인   : [(jl·nz + k)·line + i]          jl ∈ y 조각, line = Nx (주기) / 2Nx (거울)
//   x 스펙트럼: 2 × [(jl·nz + k)·Nxm + mx]     Nxm = 전체 x 모드 수
//   y 스펙트럼: 2 × [(k·nml + mxl)·Ny + jy]    nml = 로컬 x 모드 수, mxl = mx − 모드 오프셋
#include <mpi.h>

#include <memory>
#include <vector>

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::pressure {

// 한 방향 전치 하나: 송신 원소 n 은 src[send_index[n]] (× scale − subtract), 수신 원소 n 은
// dst[recv_index[n]] (거울이면 dst[recv_mirror[n]] 에도). width = 원소당 double 수 (실수 1 / 복소 2).
struct PencilExchange {
  int width = 1;
  std::vector<int> send_counts, send_displs, recv_counts, recv_displs;  // double 단위
  std::vector<int> send_index, recv_index, recv_mirror;                 // 원소 단위 (호스트 정본)
  // 백엔드 자원 (pencil_exchange.{cpp,cu}): 인덱스 표의 백엔드 사본 + 송·수신 버퍼
  int* device_send_index = nullptr;
  int* device_recv_index = nullptr;
  int* device_recv_mirror = nullptr;
  field::Buffer send_buffer, recv_buffer;
  PencilExchange() = default;
  ~PencilExchange();
  PencilExchange(const PencilExchange&) = delete;
  PencilExchange& operator=(const PencilExchange&) = delete;
};

struct PencilLayout {
  MPI_Comm comm = MPI_COMM_NULL;  // xy-평면 그룹 (같은 z 블록)
  int size = 1, rank = 0;
  int n_x = 0, n_y = 0, nz_local = 0, n_mode_x = 0;
  bool host_staging = false;      // PASCAL_TDMA_MPI_MODE=host
  // 그룹 랭크 r 의 블록 범위 (전역 내부 번호)
  std::vector<int> block_x_offset, block_x_count, block_y_offset, block_y_count;
  // x-펜슬 y 조각, y-펜슬 x 모드 조각
  std::vector<int> y_offset, y_count, mode_offset, mode_count;
  // 네 전치 — 생성자가 인덱스 표를 만들고 백엔드 자원을 잡는다
  PencilExchange block_to_lines, lines_to_block, x_to_y_spectrum, y_to_x_spectrum;

  PencilLayout(const parallel::Subdomain& sub, int n_mode_x, bool mirrored_lines);
  ~PencilLayout();
  PencilLayout(const PencilLayout&) = delete;
  PencilLayout& operator=(const PencilLayout&) = delete;

  int my_y_offset() const { return y_offset[rank]; }
  int my_y_count() const { return y_count[rank]; }
  int my_mode_offset() const { return mode_offset[rank]; }
  int my_mode_count() const { return mode_count[rank]; }
};

// ── 백엔드 (pencil_exchange.cpp = CPU 호스트 루프, pencil_exchange.cu = GPU 커널 + CUDA-aware MPI) ──
// 인덱스 표를 백엔드 메모리로 올리고 송·수신 버퍼를 잡는다 (생성 시 한 번).
void prepare_exchange(PencilExchange& exchange);
void release_exchange(PencilExchange& exchange);
// src(백엔드 메모리) → [gather × scale − subtract] → MPI_Alltoallv → [scatter] → dst(백엔드 메모리).
// dst 의 인덱스 표에 없는 원소는 건드리지 않는다 (장의 ghost 등).
void run_exchange(PencilExchange& exchange, const real_t* src, real_t* dst, MPI_Comm comm,
                  bool host_staging, real_t scale = 1, real_t subtract = 0);

}  // namespace mpmstd::equation::pressure
