#include "driver/restart.hpp"

namespace mpmstd::driver {
bool restart_due_cpu(int step, int every) {
  return every > 0 && step > 0 && (step % every == 0);
}
} // namespace mpmstd::driver
