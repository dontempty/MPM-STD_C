# P-0.5 — Interface Spike (FROZEN API)

Master plan: [`docs/REFACTOR_PLAN.md`](../../docs/REFACTOR_PLAN.md) (rev.2). This spike
**locks the core data-structure + communication API** so P0 can scaffold the whole
tree on stable signatures. It reuses the existing, validated host-single MPI layer
(`MpiContext`/`MpiTopology`/`Subdomain`) and proves the new free-function API on top
of it — without the virtual `Backend` or the class-based equations (both dropped in
rev.2).

## Frozen API

```cpp
// ── core/ (data = classes; CPU/GPU are SEPARATE types, rev.2 C1=b) ──
namespace mpmstd::core {
  class CpuField { CpuField(const Subdomain&, std::string);  // host buffer + shape-by-value
                   data(); at(i,j,k); n_total()/n_interior()/global_offset(); linear_index(); fill(); };
  class GpuField { GpuField(const Subdomain&, std::string);  // device buffer (cudaMalloc)
                   data(); to_device(host); to_host(host); /* same metadata */ };

  struct Bands { int bandwidth,n_sys,n_row; vector lo2,lo1,diag,up1,up2,rhs;   // [n_row x n_sys]
                 allocate(n_sys,n_row,bandwidth=1); idx(s,r); };
  struct ScalarSystem   { Bands x,y,z; along(Direction); };
  struct MomentumSystem { ScalarSystem comp[3]; component(Component); };       // U,V,W (coupling inside solve)
  struct PressureSystem { array<Transform,3> transform; };                     // Fft/Dct/Tdma per axis

  // host-single types re-exported + rank<->GPU binding (rev.2 C2)
  using MpiContext; using MpiTopology; using Subdomain;
  void bind_gpu_to_local_rank_cpu(const MpiContext&);   // no-op
  void bind_gpu_to_local_rank_gpu(const MpiContext&);   // cudaSetDevice(node_rank % ndev)

  // halo = explicit free fn around solve (rev.2 C3)
  void exchange_halo_cpu(CpuField&, const Subdomain&);
  void exchange_halo_gpu(GpuField&, const Subdomain&);  // device-to-device CUDA-aware MPI
}
// ── solve/ (common, implicit-only) ──
namespace mpmstd::solve { void solve_banded_cpu(core::Bands&); }   // rhs <- solution
```

## Resolved API decisions (the point of the spike)

1. **halo takes `Subdomain`, not `MpiTopology`** (plan §5 "mismatch #1"). The per-axis
   MPI datatypes + neighbour ranks live in `Subdomain::exchange_halo`; `MpiTopology` is
   reachable via `Subdomain::topology()` if ever needed.
2. **rank↔GPU was already half-built**: `MpiContext::node_rank()` (split by
   `MPI_COMM_TYPE_SHARED`) exists precisely for per-GPU binding. The spike adds only the
   `cudaSetDevice` free function — `1 rank = 1 GPU` (rev.2 C2). No `MpiTopology` rewrite.
3. **`CpuField` is "dumb" data**: shape (`n_total`/`n_interior`/`global_offset`) captured
   **by value** at construction; it holds **no** persistent topology reference. Operations
   that communicate take the `Subdomain` explicitly — matches "data=class, ops=free fn".
4. **Dual-build = folder-based** (`cpu/` vs `gpu/`), not the old `_cpu.cpp`/`_cuda.cpp`
   filename filtering. The function-name `_cpu`/`_gpu` suffix is the *caller's* selector.
   The combined (GPU-enabled) build uses **one compiler** (NVHPC `mpic++`) for both the
   host `cpu/` TUs and the device `gpu/` `.cu` TUs → avoids mixing MPI runtimes/ABIs.
   `make cpu` uses system `mpicxx` and is **CUDA-free**.
5. `kHaloWidth == 1` (project-wide). Halo exchange is faces-only (no edges/corners).

## Build & run

```bash
# CPU (works on the login node now):
make -C tests/spike cpu
mpirun -np 2 build/cpu/spike/halo_poc      # [PASS] all face halos correct
mpirun -np 1 build/cpu/spike/banded_poc    # [PASS] 1D Poisson max|err| ~ 5e-5

# GPU (needs nvhpc + A100 — submit to gpu01/gpu02):
sbatch tests/spike/spike_gpu.sh            # build (BACKEND=cuda gpu) + run link_gpu np=2
```

## Results

| check | status |
|---|---|
| 7 core/solve headers compile standalone (`-fsyntax-only`) | ✅ PASS |
| `halo_poc` CpuField + `exchange_halo_cpu`, np=2 & np=4, periodic | ✅ PASS |
| `banded_poc` Bands/ScalarSystem + `solve_banded_cpu`, 1D Poisson | ✅ PASS (max|err|=4.9e-5) |
| baseline `src/` + `apps/channel` unmodified, `make cpu lib` intact | ✅ PASS (병행 신축 불변식) |
| GPU dual-build LINK (NVHPC `mpic++`, `make BACKEND=cuda gpu`) | ✅ PASS (job 29046, gpu01) |
| GPU device-halo RUN, 1 rank=1 A100, CUDA-aware MPI (`exchange_halo_gpu` on 2×A100) | ✅ PASS (bonus — pre-de-risks P4) |

## What P0 inherits

The frozen signatures above. P0 scaffolds the full 6-layer tree (`core/ solve/ equation/
physics/ post/ driver/`, each `cpu/ gpu/`) + a real `mpmstd/Makefile` (this spike Makefile
is its seed) + a top-level dual `make cpu`/`make gpu` against these signatures, with the
existing `src/` kept as the regression baseline until parity.
