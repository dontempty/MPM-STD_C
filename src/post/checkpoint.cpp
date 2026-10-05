#include "post/checkpoint.hpp"

#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

// 구현 노트는 checkpoint.hpp 머리말 참조. 여기서는 루트만 파일을 만지고,
// 장 데이터는 "루트 수집(쓰기) / 루트 분배(읽기)" 로만 오간다.

namespace mpmstd::post {

namespace {

constexpr char kMagic[8] = {'M', 'P', 'M', 'S', 'T', 'D', 'C', 'P'};
constexpr int kVersion = 1;
constexpr int kNameBytes = 16;

struct BlockMeta {  // 랭크당 전역 내부 블록 (offset + 크기)
  int ox, oy, oz, lx, ly, lz;
};

void fail(const std::string& what) { throw std::runtime_error("checkpoint: " + what); }

// 랭크 로컬 내부(ghost 제외)를 연속 버퍼로 뽑는다/되돌린다.
void extract_interior(const std::vector<real_t>& local, const parallel::Subdomain& sub,
                      std::vector<real_t>& block) {
  const int h = halo_width;
  const int ty = sub.y().n_total, tz = sub.z().n_total;
  const int lx = sub.x().n_local, ly = sub.y().n_local, lz = sub.z().n_local;
  block.resize(static_cast<std::size_t>(lx) * ly * lz);
  std::size_t m = 0;
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k)
        block[m++] = local[(static_cast<std::size_t>(i + h) * ty + (j + h)) * tz + (k + h)];
}

void insert_interior(const std::vector<real_t>& block, const parallel::Subdomain& sub,
                     std::vector<real_t>& local) {
  const int h = halo_width;
  const int ty = sub.y().n_total, tz = sub.z().n_total;
  const int lx = sub.x().n_local, ly = sub.y().n_local, lz = sub.z().n_local;
  std::size_t m = 0;
  for (int i = 0; i < lx; ++i)
    for (int j = 0; j < ly; ++j)
      for (int k = 0; k < lz; ++k)
        local[(static_cast<std::size_t>(i + h) * ty + (j + h)) * tz + (k + h)] = block[m++];
}

// 전역 배열 ↔ 블록 (루트 전용)
void place_block(const std::vector<real_t>& block, const BlockMeta& b, int ngy, int ngz,
                 std::vector<real_t>& global_array) {
  std::size_t m = 0;
  for (int i = 0; i < b.lx; ++i)
    for (int j = 0; j < b.ly; ++j)
      for (int k = 0; k < b.lz; ++k)
        global_array[(static_cast<std::size_t>(b.ox + i) * ngy + (b.oy + j)) * ngz
                     + (b.oz + k)] = block[m++];
}

void slice_block(const std::vector<real_t>& global_array, const BlockMeta& b, int ngy,
                 int ngz, std::vector<real_t>& block) {
  block.resize(static_cast<std::size_t>(b.lx) * b.ly * b.lz);
  std::size_t m = 0;
  for (int i = 0; i < b.lx; ++i)
    for (int j = 0; j < b.ly; ++j)
      for (int k = 0; k < b.lz; ++k)
        block[m++] = global_array[(static_cast<std::size_t>(b.ox + i) * ngy
                                   + (b.oy + j)) * ngz + (b.oz + k)];
}

std::vector<BlockMeta> gather_metadata(const parallel::Subdomain& sub, MPI_Comm comm,
                                       int n_ranks) {
  const int mine[6] = {sub.x().offset, sub.y().offset, sub.z().offset,
                       sub.x().n_local, sub.y().n_local, sub.z().n_local};
  std::vector<int> all(static_cast<std::size_t>(6) * n_ranks, 0);
  MPI_Gather(mine, 6, MPI_INT, all.data(), 6, MPI_INT, 0, comm);
  std::vector<BlockMeta> metas(n_ranks);
  for (int r = 0; r < n_ranks; ++r)
    metas[r] = {all[6 * r + 0], all[6 * r + 1], all[6 * r + 2],
                all[6 * r + 3], all[6 * r + 4], all[6 * r + 5]};
  return metas;  // 루트만 의미 있는 값; 다른 랭크는 자기 것만 쓴다
}

void write_name(std::FILE* f, const std::string& name) {
  char buf[kNameBytes] = {0};
  std::snprintf(buf, sizeof(buf), "%s", name.c_str());
  std::fwrite(buf, 1, kNameBytes, f);
}

std::string read_name(std::FILE* f) {
  char buf[kNameBytes] = {0};
  if (std::fread(buf, 1, kNameBytes, f) != kNameBytes) fail("이름 읽기 실패");
  buf[kNameBytes - 1] = 0;
  return std::string(buf);
}

}  // namespace

void write_checkpoint(const std::string& path, const long step, const double time,
                      const std::vector<std::pair<std::string, const field::Field*>>& fields,
                      const std::vector<std::pair<std::string, double>>& scalars,
                      const parallel::Subdomain& sub) {
  MPI_Comm comm = sub.topology().cart_comm();
  int rank = 0, n_ranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);
  const int ngx = sub.x().n_global, ngy = sub.y().n_global, ngz = sub.z().n_global;
  const std::size_t n_global = static_cast<std::size_t>(ngx) * ngy * ngz;
  const auto metas = gather_metadata(sub, comm, n_ranks);

  // ── 수집: 장마다 전역 내부 배열을 루트에 조립 ──
  std::vector<std::vector<real_t>> global_fields;
  std::vector<real_t> local_host, block;
  for (const auto& [name, field] : fields) {
    (void)name;
    local_host.resize(field->size());
    field->download(local_host);
    extract_interior(local_host, sub, block);
    if (rank == 0) {
      global_fields.emplace_back(n_global);
      place_block(block, metas[0], ngy, ngz, global_fields.back());
      std::vector<real_t> other;
      for (int r = 1; r < n_ranks; ++r) {
        other.resize(static_cast<std::size_t>(metas[r].lx) * metas[r].ly * metas[r].lz);
        MPI_Recv(other.data(), static_cast<int>(other.size()), MPI_DOUBLE, r, 11, comm,
                 MPI_STATUS_IGNORE);
        place_block(other, metas[r], ngy, ngz, global_fields.back());
      }
    } else {
      MPI_Send(block.data(), static_cast<int>(block.size()), MPI_DOUBLE, 0, 11, comm);
    }
  }

  // ── 루트 쓰기: tmp → fsync → rename, EIO 는 백오프 재시도 ──
  int ok = 1;
  if (rank == 0) {
    const std::string tmp = path + ".tmp";
    ok = 0;
    const int backoff_seconds[3] = {0, 2, 10};
    for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
      if (backoff_seconds[attempt] > 0) sleep(backoff_seconds[attempt]);
      std::FILE* f = std::fopen(tmp.c_str(), "wb");
      if (!f) continue;
      bool good = true;
      const auto put = [&](const void* p, std::size_t bytes) {
        if (good && std::fwrite(p, 1, bytes, f) != bytes) good = false;
      };
      const int real_size = static_cast<int>(sizeof(real_t));
      const int nf = static_cast<int>(fields.size());
      const int ns = static_cast<int>(scalars.size());
      put(kMagic, 8);
      put(&kVersion, 4);
      put(&real_size, 4);
      put(&step, 8);
      put(&time, 8);
      put(&ngx, 4); put(&ngy, 4); put(&ngz, 4);
      put(&nf, 4);  put(&ns, 4);
      for (const auto& [name, value] : scalars) {
        if (good) write_name(f, name);
        put(&value, 8);
      }
      for (std::size_t q = 0; q < fields.size(); ++q) {
        if (good) write_name(f, fields[q].first);
        put(global_fields[q].data(), n_global * sizeof(real_t));
      }
      if (good && std::fflush(f) != 0) good = false;
      if (good && fsync(fileno(f)) != 0) good = false;
      if (std::fclose(f) != 0) good = false;
      if (good && std::rename(tmp.c_str(), path.c_str()) == 0) ok = 1;
      if (!ok) std::remove(tmp.c_str());
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
  if (!ok) fail("쓰기 실패 (3회 재시도 후): " + path);
}

void read_checkpoint(const std::string& path, long& step, double& time,
                     const std::vector<std::pair<std::string, field::Field*>>& fields,
                     const std::vector<std::pair<std::string, double*>>& scalars,
                     const parallel::Subdomain& sub) {
  MPI_Comm comm = sub.topology().cart_comm();
  int rank = 0, n_ranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);
  const int ngx = sub.x().n_global, ngy = sub.y().n_global, ngz = sub.z().n_global;
  const std::size_t n_global = static_cast<std::size_t>(ngx) * ngy * ngz;
  const auto metas = gather_metadata(sub, comm, n_ranks);

  std::FILE* f = nullptr;
  long header[6] = {0};  // step, ngx, ngy, ngz, nfields, nscalars (Bcast 용)
  if (rank == 0) {
    f = std::fopen(path.c_str(), "rb");
    if (!f) fail("열기 실패: " + path);
    char magic[8];
    int version = 0, real_size = 0, fx = 0, fy = 0, fz = 0, nf = 0, ns = 0;
    const auto get = [&](void* p, std::size_t bytes) {
      if (std::fread(p, 1, bytes, f) != bytes) fail("헤더 읽기 실패: " + path);
    };
    get(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) fail("매직 불일치: " + path);
    get(&version, 4);
    if (version != kVersion) fail("버전 불일치");
    get(&real_size, 4);
    if (real_size != static_cast<int>(sizeof(real_t))) fail("real_t 크기 불일치");
    get(&step, 8);
    get(&time, 8);
    get(&fx, 4); get(&fy, 4); get(&fz, 4);
    if (fx != ngx || fy != ngy || fz != ngz) fail("격자 크기 불일치 (덱과 파일이 다름)");
    get(&nf, 4); get(&ns, 4);
    if (nf != static_cast<int>(fields.size())) fail("장 수 불일치");
    if (ns != static_cast<int>(scalars.size())) fail("스칼라 수 불일치");
    for (const auto& [name, target] : scalars) {
      if (read_name(f) != name) fail("스칼라 이름/순서 불일치: " + name);
      get(target, 8);
    }
    header[0] = step;
  }
  MPI_Bcast(&step, 1, MPI_LONG, 0, comm);
  MPI_Bcast(&time, 1, MPI_DOUBLE, 0, comm);
  for (const auto& [name, target] : scalars) {
    (void)name;
    MPI_Bcast(target, 1, MPI_DOUBLE, 0, comm);
  }
  (void)header;

  std::vector<real_t> global_array, block, local_host;
  for (const auto& [name, field] : fields) {
    if (rank == 0) {
      if (read_name(f) != name) fail("장 이름/순서 불일치: " + name);
      global_array.resize(n_global);
      if (std::fread(global_array.data(), sizeof(real_t), n_global, f) != n_global)
        fail("장 데이터 읽기 실패: " + name);
      for (int r = 1; r < n_ranks; ++r) {
        slice_block(global_array, metas[r], ngy, ngz, block);
        MPI_Send(block.data(), static_cast<int>(block.size()), MPI_DOUBLE, r, 12, comm);
      }
      slice_block(global_array, metas[0], ngy, ngz, block);
    } else {
      block.resize(static_cast<std::size_t>(sub.x().n_local) * sub.y().n_local
                   * sub.z().n_local);
      MPI_Recv(block.data(), static_cast<int>(block.size()), MPI_DOUBLE, 0, 12, comm,
               MPI_STATUS_IGNORE);
    }
    // ghost 는 0 으로 두고 내부만 채운다 — 호출자가 자기 sync 로 재구축한다.
    local_host.assign(field->size(), real_t(0));
    insert_interior(block, sub, local_host);
    field->upload(local_host);
  }
  if (rank == 0) std::fclose(f);
}

}  // namespace mpmstd::post
