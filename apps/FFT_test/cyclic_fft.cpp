#include "cyclic_fft.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "device_loop.hpp"
#include "equation/pressure/solve_kernels.hpp"
#include "exchange.hpp"

namespace fft_test {

namespace {

#define FFTU_BEGIN(name) \
  do { if (timer) timer->begin(name); } while (0)
#define FFTU_END(name) \
  do { if (timer) timer->end(name); } while (0)

// ω_ny^{k·s} = exp(−2πi·((k·s) mod ny)/ny), k < m. 각도를 정수 나머지로 줄여 큰 ny 에서도 정확하게.
std::vector<real_t> make_twiddle(const CyclicDims& d) {
  const double two_pi = 2 * std::acos(-1.0);
  std::vector<real_t> table(2 * static_cast<std::size_t>(d.m));
  for (int k = 0; k < d.m; ++k) {
    const long long r = (static_cast<long long>(k) * d.s) % d.n_y;
    const double angle = -two_pi * static_cast<double>(r) / d.n_y;
    table[2 * k] = std::cos(angle);
    table[2 * k + 1] = std::sin(angle);
  }
  return table;
}

// send[r·P + b·c + t] = spectrum_y[b·m + t·q + r] · ω[t·q + r]    (P = B·c, 쓰는 쪽 연속)
struct TwiddlePack {
  CyclicDims d;
  const real_t* spectrum_y;
  const real_t* twiddle;
  real_t* send;
  MPMSTD_HD void operator()(const long long n) const {
    const long long packet = d.packet_elements();
    const int r = static_cast<int>(n / packet);
    const long long within = n - r * packet;
    const int t = static_cast<int>(within % d.c);
    const long long b = within / d.c;
    const int k = t * d.q + r;
    const long long p = b * d.m + k;
    const real_t xr = spectrum_y[2 * p], xi = spectrum_y[2 * p + 1];
    const real_t wr = twiddle[2 * k], wi = twiddle[2 * k + 1];
    send[2 * n] = xr * wr - xi * wi;
    send[2 * n + 1] = xr * wi + xi * wr;
  }
};

// spectrum[bb·q + src] = recv[src·P + bb]    (쓰는 쪽 연속)
struct RecvToSpectrum {
  CyclicDims d;
  const real_t* recv;
  real_t* spectrum;
  MPMSTD_HD void operator()(const long long n) const {
    const int source = static_cast<int>(n % d.q);
    const long long p = source * d.packet_elements() + n / d.q;
    spectrum[2 * n] = recv[2 * p];
    spectrum[2 * n + 1] = recv[2 * p + 1];
  }
};

// send[dst·P + bb] = spectrum[bb·q + dst]    (쓰는 쪽 연속)
struct SpectrumToSend {
  CyclicDims d;
  const real_t* spectrum;
  real_t* send;
  MPMSTD_HD void operator()(const long long n) const {
    const long long packet = d.packet_elements();
    const int destination = static_cast<int>(n / packet);
    const long long p = (n - destination * packet) * d.q + destination;
    send[2 * n] = spectrum[2 * p];
    send[2 * n + 1] = spectrum[2 * p + 1];
  }
};

// spectrum_y[b·m + k] = recv[(k mod q)·P + b·c + k div q] · conj(ω[k])    (쓰는 쪽 연속)
struct UnpackTwiddleConjugate {
  CyclicDims d;
  const real_t* recv;
  const real_t* twiddle;
  real_t* spectrum_y;
  MPMSTD_HD void operator()(const long long n) const {
    const int k = static_cast<int>(n % d.m);
    const long long b = n / d.m;
    const long long p = (k % d.q) * d.packet_elements() + b * d.c + k / d.q;
    const real_t xr = recv[2 * p], xi = recv[2 * p + 1];
    const real_t wr = twiddle[2 * k], wi = -twiddle[2 * k + 1];
    spectrum_y[2 * n] = xr * wr - xi * wi;
    spectrum_y[2 * n + 1] = xr * wi + xi * wr;
  }
};

}  // namespace

CyclicFft::CyclicFft(const CyclicLayout& layout, const std::size_t min_slot_elements)
    : layout_(layout), host_staging_(host_staging_requested()) {
  const CyclicDims& d = layout.dims();
  const std::size_t slot = std::max({static_cast<std::size_t>(2 * d.spectrum_elements()),
                                     static_cast<std::size_t>(d.line_elements()), min_slot_elements});
  slot_a_.assign(slot);
  slot_b_.assign(slot);
  slot_c_.assign(slot);
  slot_d_.assign(slot);
  const std::vector<real_t> table = make_twiddle(d);
  twiddle_.assign(table.size());
  twiddle_.upload(table);

  const long long packet = 2 * d.packet_elements();
  if (packet * d.q > INT_MAX) throw std::invalid_argument("CyclicFft: all-to-all displacement exceeds int");
  counts_.assign(d.q, static_cast<int>(packet));
  displs_.resize(d.q);
  for (int r = 0; r < d.q; ++r) displs_[r] = static_cast<int>(packet * r);

  fft_x_ = std::make_unique<BatchedRealFft>(d.n_x, static_cast<long long>(d.m) * d.nz, slot_a_.data(), slot_b_.data());
  fft_y_ = std::make_unique<BatchedComplexFft>(d.m, d.batches(), slot_c_.data());
  if (d.q > 1) fft_small_ = std::make_unique<BatchedComplexFft>(d.q, d.packet_elements(), slot_d_.data());
}

void CyclicFft::forward(mpmstd::common::PhaseTimer* timer) {
  const CyclicDims& d = layout_.dims();
  FFTU_BEGIN("fftu.fft_x_fwd");
  fft_x_->forward();
  FFTU_END("fftu.fft_x_fwd");
  FFTU_BEGIN("fftu.reorder_x_to_y");
  mpmstd::equation::pressure::reorder_x_to_y(slot_c_.data(), slot_b_.data(), d.m, d.nz, d.n_mode_x);
  FFTU_END("fftu.reorder_x_to_y");
  FFTU_BEGIN("fftu.fft_y_fwd");
  fft_y_->forward();
  FFTU_END("fftu.fft_y_fwd");
  if (d.q > 1) {
    FFTU_BEGIN("fftu.twiddle_pack_fwd");
    for_each(d.spectrum_elements(), TwiddlePack{d, slot_c_.data(), twiddle_.data(), slot_b_.data()});
    FFTU_END("fftu.twiddle_pack_fwd");
    FFTU_BEGIN("fftu.a2a_fwd");
    alltoallv_backend(slot_b_.data(), counts_, displs_, slot_a_.data(), counts_, displs_, layout_.group(), host_staging_);
    FFTU_END("fftu.a2a_fwd");
    FFTU_BEGIN("fftu.unpack_fwd");
    for_each(d.spectrum_elements(), RecvToSpectrum{d, slot_a_.data(), slot_d_.data()});
    FFTU_END("fftu.unpack_fwd");
    FFTU_BEGIN("fftu.fft_small_fwd");
    fft_small_->forward();
    FFTU_END("fftu.fft_small_fwd");
  }
  synchronize_device();
}

void CyclicFft::backward(mpmstd::common::PhaseTimer* timer) {
  const CyclicDims& d = layout_.dims();
  if (d.q > 1) {
    FFTU_BEGIN("fftu.fft_small_bwd");
    fft_small_->backward();
    FFTU_END("fftu.fft_small_bwd");
    FFTU_BEGIN("fftu.pack_bwd");
    for_each(d.spectrum_elements(), SpectrumToSend{d, slot_d_.data(), slot_b_.data()});
    FFTU_END("fftu.pack_bwd");
    FFTU_BEGIN("fftu.a2a_bwd");
    alltoallv_backend(slot_b_.data(), counts_, displs_, slot_a_.data(), counts_, displs_, layout_.group(), host_staging_);
    FFTU_END("fftu.a2a_bwd");
    FFTU_BEGIN("fftu.unpack_twiddle_bwd");
    for_each(d.spectrum_elements(), UnpackTwiddleConjugate{d, slot_a_.data(), twiddle_.data(), slot_c_.data()});
    FFTU_END("fftu.unpack_twiddle_bwd");
  }
  FFTU_BEGIN("fftu.fft_y_bwd");
  fft_y_->backward();
  FFTU_END("fftu.fft_y_bwd");
  FFTU_BEGIN("fftu.reorder_y_to_x");
  mpmstd::equation::pressure::reorder_y_to_x(slot_b_.data(), slot_c_.data(), d.m, d.nz, d.n_mode_x);
  FFTU_END("fftu.reorder_y_to_x");
  FFTU_BEGIN("fftu.fft_x_bwd");
  fft_x_->backward();
  FFTU_END("fftu.fft_x_bwd");
  synchronize_device();
}

}  // namespace fft_test
