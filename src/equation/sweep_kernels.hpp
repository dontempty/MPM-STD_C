#pragma once
// ADI 스윕의 공용 배관 (내부 전용) — TDMA 해 [sys + row·nsys] 를 전체 모양
// 배열의 내부로 되돌리는 산포 3종과, q += δ 갱신. scalar 와 momentum 의 solve
// 가 같이 쓴다 (세 번째 소비자가 생겨 equation/ 공용으로 승격 — core 규칙).
// sys 배치는 각 방정식의 system.hpp 표와 같다 (build_*_system 의 거울상).
// CPU(sweep_kernels.cpp)/GPU(sweep_kernels.cu) 쌍둥이가 구현한다.

#include "common/types.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation {

void scatter_x(real_t* out, const real_t* D, const parallel::Subdomain& sub);
void scatter_y(real_t* out, const real_t* D, const parallel::Subdomain& sub);
void scatter_z(real_t* out, const real_t* D, const parallel::Subdomain& sub);

// q += δ (내부만; δ 는 전체 모양 배열)
void add_increment(field::Field& q, const real_t* delta, const parallel::Subdomain& sub);

// 내부 셀만 [lower, upper] 로 클램프 — NOB 의 T 물리 클립.
// ⚠ ghost 는 건드리지 않는다: ghost 까지 클램프하면 mirror 벽 ghost(2v−f)가
// 깨져 벽 경계층이 왜곡된다 (MPM-STD_C CPU NOB 의 실측 버그 — GPU 만 수정됐던
// 것을 여기서는 처음부터 내부 한정으로 이식).
void clip_interior(field::Field& field_to_clip, const parallel::Subdomain& sub,
                   real_t lower, real_t upper);

}  // namespace mpmstd::equation
