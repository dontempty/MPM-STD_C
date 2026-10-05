#pragma once
// 재시작 체크포인트 (M11) — 장기 런이 선점·장애에 인질로 잡히지 않게 한다
// (plan §8 16: fig9 10h 급 런 권고사항).
//
// ── 이 클러스터의 IO 제약을 그대로 설계에 반영 ──────────────────────────────
// * MPI-IO **읽기**는 이 NFS 에서 데드락 전례 (256³·np2) → 루트 fread 후
//   슬랩 분배(MPI_Send) 만 쓴다. MPI_File_* 금지.
// * NFS 대량 **쓰기**는 간헐 EIO → tmp 에 쓰고 fsync 후 rename, 실패하면
//   백오프 재시도 (POD/storage.py 에서 검증된 패턴).
//
// ── 형식 v1 (네이티브 엔디안) ───────────────────────────────────────────────
//   magic[8]="MPMSTDCP" | int32 version=1 | int32 sizeof(real_t)
//   int64 step | double time | int32 ngx,ngy,ngz | int32 nfields | int32 nscalars
//   nscalars × { char name[16] | double value }
//   nfields  × { char name[16] | real_t data[ngx·ngy·ngz] }   // (i·ngy+j)·ngz+k
// 전역 내부(ghost 제외)만 저장한다 — 재시작 뒤 ghost 는 앱이 자기 sync 로
// 재구축한다 (IC 직후와 같은 호출). 분해(np)가 달라져도 읽을 수 있다.
//
// 양쪽 백엔드 공용 단일 TU: Field::download/upload 로만 장을 만지므로 .cu
// 쌍둥이가 없다 (GPU 빌드도 이 .cpp 를 담는다).
//
// 사용 (모든 랭크가 같이 부른다):
//   post::write_checkpoint(path, step, t, {{"U",&U},...}, {{"mean_dPdx",g}}, sub);
//   post::read_checkpoint(path, step, t, {{"U",&U},...}, {{"mean_dPdx",&g}}, sub);
// 읽기는 이름·순서가 쓸 때와 같아야 한다 (다르면 예외 — 같은 앱이 쓰고 읽는 전제).

#include <string>
#include <utility>
#include <vector>

#include "common/types.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::post {

void write_checkpoint(const std::string& path, long step, double time,
                      const std::vector<std::pair<std::string, const field::Field*>>& fields,
                      const std::vector<std::pair<std::string, double>>& scalars,
                      const parallel::Subdomain& sub);

void read_checkpoint(const std::string& path, long& step, double& time,
                     const std::vector<std::pair<std::string, field::Field*>>& fields,
                     const std::vector<std::pair<std::string, double*>>& scalars,
                     const parallel::Subdomain& sub);

}  // namespace mpmstd::post
