#include "post/checkpoint.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "grid/stretching.hpp"

// 격자 교차 warm-start 보간 — checkpoint.hpp 머리말 참조. checkpoint.cpp 와 같은
// 파일 형식(magic/헤더/전역 내부 장)을 읽되, 소스 격자가 이 런과 달라도 된다:
// 각 샘플의 물리 좌표를 staggered 위치로 복원해 trilinear 보간한다.
// x·y 주기(랩), z clamp. 같은 박스·같은 스트레칭 법칙 전제 (소스 면좌표를 소스
// 격자 수 + 이 런의 박스/gamma 로 재구성).

namespace mpmstd::post {

namespace {

constexpr char kMagic[8] = {'M', 'P', 'M', 'S', 'T', 'D', 'C', 'P'};
constexpr int kNameBytes = 16;

void fail(const std::string& w) { throw std::runtime_error("checkpoint_interp: " + w); }

// 소스 축의 셀중심·면 좌표 (같은 박스·스트레칭; x·y 균일, z 는 tanh/균일).
struct AxisCoords {
  std::vector<double> center;  // 길이 n
  std::vector<double> face;    // 길이 n (면 i = 셀 i 의 마이너스 면)
  double length;
};

AxisCoords source_axis(int n, double length, bool stretched, double gamma) {
  const std::vector<real_t> faces =
      stretched ? grid::make_tanh_faces(n, static_cast<real_t>(length),
                                        static_cast<real_t>(gamma))
                : grid::make_uniform_faces(n, static_cast<real_t>(length));  // 길이 n+1
  AxisCoords a;
  a.length = length;
  a.center.resize(n);
  a.face.resize(n);
  for (int i = 0; i < n; ++i) {
    a.face[i] = faces[i];
    a.center[i] = 0.5 * (faces[i] + faces[i + 1]);
  }
  return a;
}

// 1D 보간 가중: target 좌표 p 를 소스 좌표열 coord(오름차순) 에 끼운다.
// periodic=true: 주기 L 로 랩 (x·y). false: 양 끝 clamp (z 벽).
struct Weight {
  int i0, i1;
  double w;  // value = (1-w)*src[i0] + w*src[i1]
};

Weight locate(double p, const std::vector<double>& coord, double length, bool periodic) {
  const int n = static_cast<int>(coord.size());
  if (periodic) {
    // 주기 축: coord 는 [0,L) 안의 n 개 샘플. p 를 [coord[0], coord[0]+L) 로 랩.
    double pw = std::fmod(p - coord[0], length);
    if (pw < 0) pw += length;
    pw += coord[0];
    // 마지막 샘플과 첫 샘플(+L) 사이 구간까지 포함해 bracket
    for (int i = 0; i < n; ++i) {
      const double lo = coord[i];
      const double hi = (i + 1 < n) ? coord[i + 1] : coord[0] + length;
      if (pw >= lo - 1e-12 && pw <= hi + 1e-12) {
        const double w = (hi > lo) ? (pw - lo) / (hi - lo) : 0.0;
        return {i, (i + 1) % n, std::min(1.0, std::max(0.0, w))};
      }
    }
    return {0, 0, 0.0};  // 도달 불가
  }
  // clamp 축
  if (p <= coord[0]) return {0, 0, 0.0};
  if (p >= coord[n - 1]) return {n - 1, n - 1, 0.0};
  for (int i = 0; i + 1 < n; ++i)
    if (p <= coord[i + 1]) {
      const double w = (p - coord[i]) / (coord[i + 1] - coord[i]);
      return {i, i + 1, w};
    }
  return {n - 1, n - 1, 0.0};
}

}  // namespace

void read_checkpoint_interpolated(
    const std::string& path, long& step, double& time,
    const std::vector<std::tuple<std::string, field::Field*, Stagger>>& fields,
    const std::vector<std::pair<std::string, double*>>& scalars,
    const grid::Grid& grid, const parallel::Subdomain& sub, const double lx,
    const double ly, const double lz, const bool uniform_z, const double gamma_z) {
  MPI_Comm comm = sub.topology().cart_comm();
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // ── 헤더 (루트) ──
  std::FILE* f = nullptr;
  int src_nx = 0, src_ny = 0, src_nz = 0, nf = 0, ns = 0;
  if (rank == 0) {
    f = std::fopen(path.c_str(), "rb");
    if (!f) fail("열기 실패: " + path);
    char magic[8];
    int version = 0, real_size = 0;
    const auto get = [&](void* p, std::size_t b) {
      if (std::fread(p, 1, b, f) != b) fail("헤더 읽기 실패");
    };
    get(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) fail("매직 불일치");
    get(&version, 4);
    get(&real_size, 4);
    if (real_size != static_cast<int>(sizeof(real_t))) fail("real_t 크기 불일치");
    get(&step, 8);
    get(&time, 8);
    get(&src_nx, 4);
    get(&src_ny, 4);
    get(&src_nz, 4);
    get(&nf, 4);
    get(&ns, 4);
    for (const auto& [name, target] : scalars) {
      char buf[kNameBytes];
      if (std::fread(buf, 1, kNameBytes, f) != kNameBytes) fail("스칼라 이름 읽기 실패");
      buf[kNameBytes - 1] = 0;
      if (name != buf) fail("스칼라 이름/순서 불일치: " + name);
      get(target, 8);
    }
    if (nf != static_cast<int>(fields.size())) fail("장 수 불일치");
    if (ns != static_cast<int>(scalars.size())) fail("스칼라 수 불일치");
  }
  long step_b = step;
  MPI_Bcast(&step_b, 1, MPI_LONG, 0, comm);
  step = step_b;
  MPI_Bcast(&time, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&src_nx, 1, MPI_INT, 0, comm);
  MPI_Bcast(&src_ny, 1, MPI_INT, 0, comm);
  MPI_Bcast(&src_nz, 1, MPI_INT, 0, comm);
  for (const auto& [name, target] : scalars) {
    (void)name;
    MPI_Bcast(target, 1, MPI_DOUBLE, 0, comm);
  }

  // ── 소스 축 좌표 (같은 박스·스트레칭) ──
  const AxisCoords sx = source_axis(src_nx, lx, false, 0.0);
  const AxisCoords sy = source_axis(src_ny, ly, false, 0.0);
  const AxisCoords sz = source_axis(src_nz, lz, !uniform_z, gamma_z);
  const std::size_t src_n =
      static_cast<std::size_t>(src_nx) * src_ny * src_nz;
  std::vector<real_t> src(src_n);

  const int h = halo_width;
  const int lxn = sub.x().n_local, lyn = sub.y().n_local, lzn = sub.z().n_local;
  const int tyt = sub.y().n_total, tzt = sub.z().n_total;

  for (const auto& [name, field_ptr, stag] : fields) {
    // 루트: 장 이름 확인 + 전역 소스장 읽기 → 전체 Bcast
    if (rank == 0) {
      char buf[kNameBytes];
      if (std::fread(buf, 1, kNameBytes, f) != kNameBytes) fail("장 이름 읽기 실패");
      buf[kNameBytes - 1] = 0;
      if (name != buf) fail("장 이름/순서 불일치: " + name);
      if (std::fread(src.data(), sizeof(real_t), src_n, f) != src_n)
        fail("장 데이터 읽기 실패: " + name);
    }
    MPI_Bcast(src.data(), static_cast<int>(src_n),
              sizeof(real_t) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT, 0, comm);

    // 소스 좌표열: staggering 에 따라 center vs face 선택
    const std::vector<double>& cx = (stag == Stagger::FaceX) ? sx.face : sx.center;
    const std::vector<double>& cy = (stag == Stagger::FaceY) ? sy.face : sy.center;
    const std::vector<double>& cz = (stag == Stagger::FaceZ) ? sz.face : sz.center;
    const bool px = true, py = true, pz = false;  // x·y 주기, z 벽

    std::vector<real_t> local(field_ptr->size(), real_t(0));
    // 이 런(타깃) 격자의 staggered 물리 좌표 — 로컬 내부 셀 q → grid 축의 q+h
    for (int i = 0; i < lxn; ++i) {
      const double tx = (stag == Stagger::FaceX) ? grid.x.face[i + h]
                                                 : grid.x.center[i + h];
      const Weight wx = locate(tx, cx, lx, px);
      for (int j = 0; j < lyn; ++j) {
        const double ty = (stag == Stagger::FaceY) ? grid.y.face[j + h]
                                                   : grid.y.center[j + h];
        const Weight wy = locate(ty, cy, ly, py);
        for (int k = 0; k < lzn; ++k) {
          const double tz = (stag == Stagger::FaceZ) ? grid.z.face[k + h]
                                                     : grid.z.center[k + h];
          const Weight wz = locate(tz, cz, lz, pz);
          // 8점 trilinear
          const auto S = [&](int a, int b, int c) -> double {
            return src[(static_cast<std::size_t>(a) * src_ny + b) * src_nz + c];
          };
          const double c00 = (1 - wx.w) * S(wx.i0, wy.i0, wz.i0) + wx.w * S(wx.i1, wy.i0, wz.i0);
          const double c01 = (1 - wx.w) * S(wx.i0, wy.i0, wz.i1) + wx.w * S(wx.i1, wy.i0, wz.i1);
          const double c10 = (1 - wx.w) * S(wx.i0, wy.i1, wz.i0) + wx.w * S(wx.i1, wy.i1, wz.i0);
          const double c11 = (1 - wx.w) * S(wx.i0, wy.i1, wz.i1) + wx.w * S(wx.i1, wy.i1, wz.i1);
          const double c0 = (1 - wy.w) * c00 + wy.w * c10;
          const double c1 = (1 - wy.w) * c01 + wy.w * c11;
          const double val = (1 - wz.w) * c0 + wz.w * c1;
          local[(static_cast<std::size_t>(i + h) * tyt + (j + h)) * tzt + (k + h)] =
              static_cast<real_t>(val);
        }
      }
    }
    field_ptr->upload(local);
  }
  if (rank == 0) std::fclose(f);
}

}  // namespace mpmstd::post
