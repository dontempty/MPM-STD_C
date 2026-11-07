#include "cyclic_layout.hpp"

#include <stdexcept>
#include <string>

namespace fft_test {

CyclicLayout::CyclicLayout(const mpmstd::parallel::Subdomain& sub) {
  const auto& topo = sub.topology();
  // xy 그룹 = 같은 z 블록 (기존 PlaneLayout 과 같은 분할·순서). z 줄 위의 랭크들은 같은 그룹 랭크 = 같은 모드를 갖는다.
  const int group_key = topo.y().rank * topo.x().nprocs + topo.x().rank;
  MPI_Comm_split(topo.cart_comm(), topo.z().rank, group_key, &group_);
  int size = 1, rank = 0;
  MPI_Comm_size(group_, &size);
  MPI_Comm_rank(group_, &rank);

  const int n_x = sub.x().n_global, n_y = sub.y().n_global;
  if (n_y % (size * size) != 0)
    throw std::invalid_argument("CyclicLayout: need q^2 | ny, got q = npx*npy = " + std::to_string(size)
                                + ", ny = " + std::to_string(n_y));
  dims_.n_x = n_x;
  dims_.n_y = n_y;
  dims_.n_mode_x = n_x / 2 + 1;
  dims_.q = size;
  dims_.s = rank;
  dims_.m = n_y / size;
  dims_.c = dims_.m / size;
  dims_.nz = sub.z().n_local;
}

CyclicLayout::~CyclicLayout() {
  if (group_ != MPI_COMM_NULL) MPI_Comm_free(&group_);
}

}  // namespace fft_test
