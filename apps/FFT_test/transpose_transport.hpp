#pragma once
// MPM-STD 압력 경로의 통신 방식 시험 (plan.md §11·§12) — 기존 소스를 고치지 않는다.
//
// src/equation/pressure/pencil_exchange.cu 는 MPI_Alltoallv 를 부른다. 이 앱은 MPI 표준 프로파일링 인터페이스(PMPI)로
// MPI_Alltoallv 를 가로채, 등록한 통신자에서만 다른 방식으로 보낸다. 등록하지 않은 통신자와 alltoallv 는
// PMPI_Alltoallv 그대로다. 이 앱의 실행파일에만 적용된다.
//
//   FFT_TEST_TRANSPOSE
//     alltoallv (기본)  PMPI_Alltoallv — 지금의 MPM-STD
//     pairwise          단계 i 마다 me+i 에게 보내고 me−i 에게서 받은 뒤 끝나기를 기다린다 (MPI_Isend·MPI_Irecv·MPI_Waitall)
//     nccl              ncclSend·ncclRecv 묶음 (ncclGroupStart/End) 후 동기화. 불균등 분할도 되게 ncclAlltoAll 은 쓰지 않는다.
//   FFT_TEST_TDMA_A2A     z TDMA 축소계의 MPI_Alltoallv (linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cuda.cu)
//     alltoallv (기본) | pairwise
//   모르는 값이면 throw (조용히 기본값으로 돌지 않는다). pairwise 는 자기 몫이 빠진 호출만 받는다 (MPMSTD_A2A_SELF 기본값 copy).
//
// 멀티노드 NCCL 은 사이트 규정대로 NCCL_PROTO=^LL128 과 aws-ofi-nccl 모듈이 필요하다 (job_transport.sh).

#include <mpi.h>

namespace fft_test {

const char* transpose_mode_name();
const char* tdma_mode_name();
// 전치 통신자를 등록한다. nccl 이면 NCCL 통신자를 만든다 (집합 호출, cudaSetDevice 뒤에).
void register_transpose_comm(MPI_Comm comm);
// z TDMA 플랜의 통신자를 등록한다 (플랜이 dup 한 사본, plan_z.backend.comm).
void register_tdma_comm(MPI_Comm comm);
// NCCL 통신자를 정리한다. MPI_Finalize 전에 부른다.
void release_transpose_comms();

}  // namespace fft_test
