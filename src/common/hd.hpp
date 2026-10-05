#pragma once
// CPU/GPU 공유 수식 함수 표식 — MPM-STD_C 의 MPMSTD_HD 관례 그대로 (설계 규칙 D3:
// 수식은 한 곳에만. CPU .cpp 루프와 GPU .cu 커널이 같은 인라인 함수를 부른다.
// 성공 사례 = MPM-STD_C 의 scalar_var_stencil.hpp — 그 패턴이 여기의 표준이다.)
//
// GPU 빌드(nvc++ -cuda)에서는 __host__ __device__ 가 되고, CPU 빌드(g++)에서는
// 사라진다. 커널에 들어가는 함수는 "스칼라·POD 뷰만 받는 순수 함수"여야 한다.

#if defined(__CUDACC__) || defined(__NVCOMPILER_CUDA__)
#define MPMSTD_HD __host__ __device__
#else
#define MPMSTD_HD
#endif
