#pragma once
// Field 의 백엔드 선택 — 프로젝트 전체의 유일한 CPU/GPU 분기 스위치는
// MPMSTD_GPU 하나다 (설계 규칙 D5). 모든 코드는 이 헤더만 include 하고,
// 두 백엔드의 Field 는 같은 이름·같은 공개 API 를 가진다 (D4):
//   * CPU(field/cpu/field.hpp): std::vector 저장, 호스트 원소 접근 operator() 제공
//   * GPU(field/gpu/field.hpp): cudaMalloc 디바이스 상주 — **호스트 원소 접근이
//     아예 없다** (컴파일 에러로 강제). 계산은 커널로만, 호스트와는
//     upload()/download() 로만 오간다 (2026-10-04 사용자 지시: 배열은 GPU 에
//     올려 두고 통계 출력 때만 CPU 로).

#ifdef MPMSTD_GPU
#include "field/gpu/field.hpp"
#else
#include "field/cpu/field.hpp"
#endif
