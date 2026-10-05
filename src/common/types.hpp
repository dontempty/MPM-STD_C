#pragma once
// 프로젝트 전역 타입·상수.
namespace mpmstd {

// double 고정 — PaScaL_TDMA 와 cuFFT(Z2Z) 가 double 전용이라 단정도 빌드는
// 어차피 불가능하다 (MPM-STD_C 에서 static_assert 로 막혀 있었고 실제로 쓰인
// 적도 없다). 필요해지면 그때 일반화한다.
using real_t = double;

// ghost 층 두께. 모든 스텐실이 컴팩트 3점(±1)이라 1 로 충분하다.
// 배열 치수는 전부 n_total = n_local + 2*halo_width 로 이 값에서 파생된다.
inline constexpr int halo_width = 1;

}  // namespace mpmstd
