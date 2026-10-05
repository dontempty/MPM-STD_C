#include "equation/sweep_kernels.hpp"

#include <cstddef>

namespace mpmstd::equation {

namespace {

inline std::size_t at(const int i, const int j, const int k, const int ny_total,
                      const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

}  // namespace

void scatter_x(real_t* out, const real_t* D, const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i2 * i3;
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k)
        out[at(i, j, k, sub.y().n_total, sub.z().n_total)] =
            D[static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h)];
}

void scatter_y(real_t* out, const real_t* D, const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i3;
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k)
        out[at(i, j, k, sub.y().n_total, sub.z().n_total)] =
            D[static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h)];
}

void scatter_z(real_t* out, const real_t* D, const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i2;
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k)
        out[at(i, j, k, sub.y().n_total, sub.z().n_total)] =
            D[static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h)];
}

void add_increment(field::Field& q_field, const real_t* delta,
                   const parallel::Subdomain& sub) {
  const int h = halo_width;
  const field::View q = q_field.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        q(i, j, k) += delta[at(i, j, k, sub.y().n_total, sub.z().n_total)];
}

void clip_interior(field::Field& field_to_clip, const parallel::Subdomain& sub,
                   const real_t lower, const real_t upper) {
  const int h = halo_width;
  const field::View f = field_to_clip.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        if (f(i, j, k) < lower) f(i, j, k) = lower;
        if (f(i, j, k) > upper) f(i, j, k) = upper;
      }
}

}  // namespace mpmstd::equation
