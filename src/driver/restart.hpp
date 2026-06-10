#pragma once

namespace mpmstd::driver {

// Case-shared restart orchestration (rev.2 M5 + U6). Decides WHEN to checkpoint;
// the actual field IO is post::write_restart_cpu / read_restart_cpu. (body P1.)
bool restart_due_cpu(int step, int every);

} // namespace mpmstd::driver
