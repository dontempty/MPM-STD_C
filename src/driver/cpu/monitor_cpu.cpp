#include "driver/monitor.hpp"

#include <cstdio>

// P0 skeleton stub. P1 wires the real divergence/CFL monitor line.
namespace mpmstd::driver {
void print_monitor_cpu(const core::MpiContext& mpi, int step, real_t time, real_t dt, real_t div_max) {
  if (mpi.is_root())
    std::printf("step=%d time=%.6g dt=%.3g div=%.3e\n", step, double(time), double(dt), double(div_max));
}
} // namespace mpmstd::driver
