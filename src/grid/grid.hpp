#pragma once

#include "common/types.hpp"
#include "common/direction.hpp"
#include "common/macros.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi/subdomain.hpp"

#include <vector>
#include <array>

namespace mpmstd::grid {

// Configuration for a single axis.
struct AxisConfig {
  int          n_global = 0;
  real_t       length   = 1.0;
  StretchKind  stretch  = StretchKind::Uniform;
  real_t       gamma    = 0.0;
};

// Staggered grid on a structured rectangular domain.
//
// For each axis d we store **local** (subdomain-restricted) arrays:
//   xc[d][i]    : cell-center coordinate (i = 0..n_total[d]-1, halo included)
//   xf[d][i]    : face coordinate (i = 0..n_total[d], i.e. one more than xc)
//   dx[d][i]    : cell width = xf[d][i+1] - xf[d][i]              (size n_total[d])
//   dmx[d][i]   : face-to-face distance centered at xc[d][i]      (size n_total[d])
//
// These arrays follow PaScaL_TCS conventions but live per-rank: halo cells
// (index 0 and n_total[d]-1) get extrapolated coordinates.
//
// `xc` is what stencils use for cell-centered quantities (P, T). `xf` is
// the location of velocity components on their respective faces.

class Grid {
public:
  Grid(const parallel::mpi::Subdomain& sub,
       std::array<AxisConfig, 3>       axes);

  const parallel::mpi::Subdomain& subdomain() const { return sub_; }

  // Per-axis accessors.
  const std::vector<real_t>& xc (Direction d) const { return xc_ [to_int(d)]; }
  const std::vector<real_t>& xf (Direction d) const { return xf_ [to_int(d)]; }
  const std::vector<real_t>& dx (Direction d) const { return dx_ [to_int(d)]; }
  const std::vector<real_t>& dmx(Direction d) const { return dmx_[to_int(d)]; }

  // Global domain length along axis d.
  real_t length(Direction d) const { return axes_[to_int(d)].length; }
  int    n_global(Direction d) const { return axes_[to_int(d)].n_global; }
  StretchKind stretch(Direction d) const { return axes_[to_int(d)].stretch; }

  // Raw pointers — kernels prefer these.
  const real_t* xc_ptr (Direction d) const { return xc_ [to_int(d)].data(); }
  const real_t* dx_ptr (Direction d) const { return dx_ [to_int(d)].data(); }
  const real_t* dmx_ptr(Direction d) const { return dmx_[to_int(d)].data(); }

private:
  const parallel::mpi::Subdomain& sub_;
  std::array<AxisConfig, 3>       axes_;

  std::array<std::vector<real_t>, 3> xc_;
  std::array<std::vector<real_t>, 3> xf_;
  std::array<std::vector<real_t>, 3> dx_;
  std::array<std::vector<real_t>, 3> dmx_;

  void build_axis_(int axis);
};

} // namespace mpmstd::grid
