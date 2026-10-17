#pragma once
// 격자 — "전역 면 좌표를 먼저 만들고, 이 랭크의 몫을 잘라낸다".
// 스트레칭이 전역 좌표계에서 한 번 계산되므로 랭크 수를 바꿔도 격자가 비트
// 단위로 동일하다. 이것이 분해 독립 재시작의 전제다 (test_grid 가 보증).
//
// 축 추상화는 없다 (2026-10-03 사용자 지시): Grid 는 그냥 세 축 x, y, z 의
// 묶음이고, 각 축은 make_axis() 가 독립적으로 만든다. 커널은
// grid.z.cell_width[k] 처럼 축을 이름으로 집어 읽는다.
//
// 한 축의 레이아웃 (모든 배열은 로컬, ghost 포함):
//
//   face[k]            k = 0 .. n_total     면 좌표 — center 보다 1개 많다
//   center[i]          i = 0 .. n_total-1   셀 중심 = (face[i] + face[i+1]) / 2
//   cell_width[i]      = face[i+1] - face[i]
//   center_distance[i] = center[i] - center[i-1]
//                        ([0] 은 center[-1] 이 없으므로 cell_width[0] — PaScaL 외삽 관례)
//
//   ghost 셀 = 인덱스 0 과 n_total-1.  로컬 i ↔ 전역 offset + (i - halo_width).
//   전역 범위 밖의 ghost 면 좌표는 끝 셀 폭의 거울 반사로 외삽된다.
//
// 이름 주의: center_distance 는 이름 그대로 "이웃 셀 중심 사이 거리"다.
// (MPM-STD_C 의 dmx 는 헤더 주석이 "face-to-face"라고 잘못 적혀 있었다 —
//  구현은 동일하다. 이름으로 그 혼동을 없앤다.)
//
// Grid 는 MPI 를 모른다 — 랭크 분할은 AxisRange 값으로만 전달받는다.
// (M3 의 subdomain 이 이 값을 채우고, 단일 프로세스는 {n, 0, n} 이다.)

#include <string>
#include <vector>

#include "common/types.hpp"
#include "field/buffer.hpp"

namespace mpmstd::grid {

// 이 랭크가 담당하는 한 축의 전역 셀 구간.
struct AxisRange {
  int n_global = 0;  // 축 전역 내부 셀 수
  int offset = 0;    // 첫 로컬 내부 셀의 전역 번호 (0-based)
  int n_local = 0;   // 로컬 내부 셀 수
};

// 한 축의 격자 — 순수 데이터. make_axis() 가 채운다.
struct Axis1D {
  int n_global = 0;
  int offset = 0;
  int n_local = 0;
  int n_total = 0;    // = n_local + 2*halo_width (배열 치수)
  real_t length = 0;  // 전역 도메인 길이

  std::vector<real_t> face;             // 크기 n_total + 1
  std::vector<real_t> center;           // 크기 n_total
  std::vector<real_t> cell_width;       // 크기 n_total
  std::vector<real_t> center_distance;  // 크기 n_total
  // 역수 — 스텐실의 "/ dx", "/ dmx" 를 곱셈으로 (2026-10-17, 운동량 RHS 최적화 1: 셀당 FP64 나눗셈 ~60회 제거).
  // a/b 와 a·(1/b) 는 마지막 비트가 다를 수 있다 → 이 배열을 쓰는 커널은 비트 동일이 아니라 허용오차 게이트.
  std::vector<real_t> inverse_cell_width;       // 1 / cell_width
  std::vector<real_t> inverse_center_distance;  // 1 / center_distance

#ifdef MPMSTD_GPU
  // 커널이 읽는 메트릭의 디바이스 사본 — make_axis 가 한 번 올린다.
  // (격자는 불변이라 복사 한 번이면 끝; 호스트 배열은 통계·출력용으로 유지)
  field::Buffer device_cell_width;
  field::Buffer device_center_distance;
  field::Buffer device_inverse_cell_width;
  field::Buffer device_inverse_center_distance;
#endif
};

// 한 축을 만든다. global_faces 는 길이 n_global+1 (stretching.hpp 산출물),
// axis_name 은 에러 메시지용("x"/"y"/"z"). 잘못된 입력(크기 불일치, 비단조 면,
// 범위 밖 구간)은 전부 throw.
Axis1D make_axis(const std::vector<real_t>& global_faces, AxisRange range,
                 const std::string& axis_name);

// 격자 = 세 축의 묶음. 집계 초기화로 만든다:
//   Grid grid{make_axis(faces_x, range_x, "x"),
//             make_axis(faces_y, range_y, "y"),
//             make_axis(faces_z, range_z, "z")};
struct Grid {
  Axis1D x;
  Axis1D y;
  Axis1D z;
};

}  // namespace mpmstd::grid
