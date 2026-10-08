#include "post/spectra_kernels.hpp"

#include <algorithm>

namespace mpmstd::post {

void accumulate_line_power(real_t* power, const real_t* spectrum, const int n_lines,
                           const int n_mode, const int n, const real_t scale) {
  for (int line = 0; line < n_lines; ++line)
    for (int m = 0; m < n_mode; ++m) {
      const std::size_t c = 2 * (static_cast<std::size_t>(line) * n_mode + m);
      const real_t re = spectrum[c], im = spectrum[c + 1];
      const real_t one_sided = (m == 0 || (n % 2 == 0 && m == n / 2)) ? real_t(1) : real_t(2);
      power[static_cast<std::size_t>(line) * n_mode + m] += scale * one_sided * (re * re + im * im);
    }
}

void zero_buffer(real_t* data, const std::size_t n) { std::fill(data, data + n, real_t(0)); }

void gather_lines(real_t* staging, const real_t* in, const int length, const int n_lines,
                  const long long stride) {
  for (int line = 0; line < n_lines; ++line)
    for (int i = 0; i < length; ++i)
      staging[static_cast<std::size_t>(line) * length + i] = in[i * stride + line];
}

}  // namespace mpmstd::post
