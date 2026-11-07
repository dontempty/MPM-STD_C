#pragma once
// 백엔드 버퍼끼리의 MPI_Alltoallv (counts·displs 는 double 단위).
//   GPU: 디바이스 포인터를 그대로 MPI 에 (GPU-aware cray-mpich, MPICH_GPU_SUPPORT_ENABLED=1).
//        자기 몫은 cudaMemcpy 로 (GTL 의 디바이스 self 전송이 느린 전례 — pencil_exchange.cu 와 같은 처리).
//        FFT_TEST_MPI_MODE=host 면 호스트 경유 (대조용).
//        FFT_TEST_A2A 로 상대 간 전송 방식을 고른다 (통신 방식 시험, plan.md §11):
//          alltoallv (기본)  MPI_Alltoallv — MPM-STD 전치와 같다
//          p2p               상대마다 MPI_Isend·MPI_Irecv 후 MPI_Waitall — cuDecomp mpi_p2p 계열과 같은 호출
//          pairwise          MPI_Sendrecv 로 한 번에 한 상대씩 (단계 i: me+i 에게 보내고 me−i 에게서 받는다)
//        모르는 값이면 throw (조용히 기본값으로 돌지 않는다).
//   CPU: 그대로 MPI_Alltoallv.
// 호출 전에 디바이스 작업을 동기화한다 (팩 커널 완료 보장).

#include <mpi.h>

#include <vector>

#include "common/types.hpp"

namespace fft_test {

using mpmstd::real_t;

bool host_staging_requested();
const char* a2a_mode_name();   // FFT_TEST_A2A 를 검사해 돌려준다 (로그·CSV 머리용)

void alltoallv_backend(const real_t* send, const std::vector<int>& send_counts,
                       const std::vector<int>& send_displs, real_t* recv,
                       const std::vector<int>& recv_counts, const std::vector<int>& recv_displs,
                       MPI_Comm comm, bool host_staging);

}  // namespace fft_test
