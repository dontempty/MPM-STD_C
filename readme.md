# MPM-STD_C

**MPM-STD_C is my own multi-GPU DNS solver** for **wall-bounded thermal convection** — a unified **C++/CUDA** incompressible-flow code I designed and implemented from the ground up. One source tree produces both **CPU (MPI)** and **GPU (CUDA + MPI)** binaries through a folder-selected "dual build", and the GPU path scales across multiple A100s via z-split CUDA-aware MPI.

It solves two canonical DNS benchmarks — **DHVC** (Differentially-Heated Vertical Channel) and **RBC** (Rayleigh–Bénard Convection, both constant-property OB and variable-property NOB) — plus auxiliary **channel**, **cavity**, and a **validate** harness.

> Numerics are validated against the Fortran reference solver **PaScaL_TCS** and published DNS (Ng et al., Kim et al. 2023). PaScaL_TCS is only the correctness yardstick; all code here is my own implementation.

This README is a **project map**: read it top-to-bottom to understand the whole codebase — how it's structured, how a timestep flows through the modules, and what every directory and file does.

---

## Table of contents
1. [The big picture](#1-the-big-picture)
2. [Physics & numerical method](#2-physics--numerical-method)
3. [Conventions you must know (gotchas)](#3-conventions-you-must-know-gotchas)
4. [Project structure at a glance](#4-project-structure-at-a-glance)
5. [How a simulation runs (execution flow)](#5-how-a-simulation-runs-execution-flow)
6. [Component reference — every directory & file](#6-component-reference--every-directory--file)
7. [Applications (`apps/`)](#7-applications-apps)
8. [Building](#8-building)
9. [Running](#9-running)
10. [Restart & cross-mesh warm-start](#10-restart--cross-mesh-warm-start)
11. [Tests, reports, docs](#11-tests-reports-docs)

---

## 1. The big picture

The code is organized as **one static library** (`libmpmstd.a`, built from `src/`) plus thin **application drivers** (`apps/*/main*.{cpp,cu}`). Each app wires the library modules into a time loop for a specific physics case.

The library is built from **six numeric layers** stacked on **host infrastructure**:

```
            apps/ (dhvc, rbc, channel, cavity, validate)  ← thin drivers: config → setup → time loop
            ───────────────────────────────────────────
 numeric    driver/      cfl (adaptive dt), monitor, restart scheduling
 layers     post/        statistics, restart I/O, cross-mesh warm-start, diagnostics
            physics/     buoyancy, properties (NOB), forcing, (ibm/les stubs)
            equation/    momentum · scalar(energy) · pressure   ← the PDE solvers
            solve/       generic banded + Poisson orchestration (earlier path)
            core/        Domain, Subdomain, Field (cpu/gpu), Variables, Systems, halo, BC ops
            ───────────────────────────────────────────
 host       grid/        stretched MAC grid (xc/xf/dx/dmx)
 infra      boundary/    Problem / boundary conditions from TOML
            linear_solver/  PaScaL_TDMA registry (cpu + cuda backends)
            parallel/    MPI topology/subdomain, CUDA helpers, backend abstraction
            config/      TOML loader + logger
            common/      types (real_t), Direction enum, macros (kHaloWidth)
```

**Dual build.** Backend is selected by **folder**: every numeric layer has a `cpu/` (host `.cpp`) and a `gpu/` (`.cu`) subfolder implementing the same functions with `_cpu` / `_gpu` suffixes. `make BACKEND=cpu` compiles `cpu/` + backend-agnostic code; `make BACKEND=cuda` additionally compiles the `gpu/` `.cu`. The caller picks the backend by calling `foo_cpu(...)` vs `foo_gpu(...)`. CPU and GPU paths mirror each other.

---

## 2. Physics & numerical method

**Equations.** Incompressible Navier–Stokes + energy with Boussinesq buoyancy (OB) or full variable-property coupling (NOB).

**Non-dimensionalization — free-fall scaling.** Velocity = free-fall velocity `U_ff = √(gβΔT H)`, length = channel width `H` → **1 time unit = 1 free-fall time** `H/U_ff`. Hence `nu = √(Pr/Ra)`, `alpha = 1/√(Ra·Pr)`. Every monitor/stats quantity (`t, dt, Re, KE, Nu, Trms, …`) is in these units.

**Per-timestep scheme.**
- **Momentum** — Beam–Warming **delta-form ADI** (3 sweeps): own-direction diffusion implicit ×2, cross ×1; **semi-implicit convection** (0.25 explicit RHS + 0.25 implicit bands). After the sweeps, **block lower-triangular velocity coupling** corrects the increments (cross-velocity convection of δU/δV/δW).
- **Scalar (temperature)** — ADI; OB = explicit convection + Crank–Nicolson diffusion; NOB = semi-implicit var-κ kernels.
- **Pressure** — **incremental (van Kan) projection**: predictor subtracts `dt·∇P_acc`; a Poisson solve gives δp; velocity is projected divergence-free; `P_acc += δp`. The Poisson solve = **pencil FFT in (x,y) + distributed TDMA in z** (FFTW/cuFFT + PaScaL_TDMA).
- **OB vs NOB** — the `*_var` (variable-property) kernels are the faithful scheme; at `ΔT=0` the property model → unit constants and they reduce **exactly** to OB. DHVC runs `*_var` at unit properties; dedicated `*_ob` kernels exist but are dead for DHVC (live only for `rbc_gpu_ob`).

**MAC staggered grid.** All five fields share one shape `[n_total[0]][n_total[1]][n_total[2]]` (C-order, k fastest, halo `kHaloWidth = 1`). Staggering is an indexing convention:

| Field | X | Y | Z |
|---|---|---|---|
| U | **face** | center | center |
| V | center | **face** | center |
| W | center | center | **face** |
| P, T | center | center | center |

Per-axis coordinates: `xc` (center), `xf` (face), `dx` (cell width), `dmx` (face-to-face metric for 2nd derivatives). Stretching: `uniform` (x,y) or `tanh γ` (z).

---

## 3. Conventions you must know (gotchas)

- **`z` is the wall-normal axis** (PaScaL_TCS uses `y`). Mapping the paper's `Nx×Ny×Nz` (Ny=wall, Nz=span) → our `(n1m, n2m, n3m) = (Nx, Nz_span, Ny_wall)`. Swapping wrong under-resolves the boundary layer.
- **GPU runs z-split only** (`np1=1, np2=1, np3=N`) — the fast path (CUDA-aware MPI halos, no whole-field host staging). x/y split forces slow host-staged FFT transposes.
- **MPI-only parallelism** — no OpenMP / std::thread / FFTW threads.
- **Wall BC is folded into the TDMA matrix** (zero-ghost for cell-centered U/V, not antisymmetric); the W face at the z-wall (interior index `k=h`) is zeroed explicitly after the W solve.
- **Full identifiers, no abbreviations** (`domain` not `dom`, `momentum`/`pressure` not `mom`/`poi`).
- **Restarts are decomposition-independent** — one global C-order `.bin` per field; restart with a different GPU count or mesh (§10).
- **Adaptive dt is `min(dt_cfl, dt_cap)`** — never `max(...)` (that pins dt and blows up explicit convection).

---

## 4. Project structure at a glance

```
MPM-STD_C/
├── src/                  # the library (libmpmstd.a)
│   ├── common/           # real_t, Direction, kHaloWidth, host/device macros
│   ├── core/             # Domain/Subdomain/Field/Variables/Systems + halo + BC ops (cpu/ gpu/)
│   ├── grid/             # stretched MAC grid (xc/xf/dx/dmx) + device copy
│   ├── parallel/         # MPI (context/topology/subdomain) + cuda/ helpers + backend/
│   ├── boundary/         # Problem + boundary conditions, loaded from TOML
│   ├── config/           # TOML-ish config loader + logger
│   ├── linear_solver/    # tdma/ (PaScaL_TDMA registry, cpu+cuda backends)
│   ├── solve/            # generic banded + Poisson orchestration (earlier/general path)
│   ├── equation/         # momentum/ scalar/ pressure/   ← core PDE solvers (cpu/ + gpu/)
│   ├── physics/          # buoyancy/ properties/ forcing/ (active); ibm/ les/ (stubs)
│   ├── post/             # statistics, io (restart + warm-start interp), diagnostics
│   └── driver/           # cfl (adaptive dt), monitor, restart scheduling
├── apps/                 # entry points + .toml inputs + plotting notebooks
│   ├── dhvc/  rbc/  channel/  cavity/  validate/
├── scripts/              # SLURM batch scripts (build + run + warm-start ladder)
├── tests/                # unit / integration / regression (vs PaScaL_TCS)
├── Report/   docs/       # design+status reports; specs (NOB, progress, refactor)
├── backup/               # archived prior results
├── Makefile  Makefile.inc   # dual (cpu/cuda) build
└── OPTIMIZATIONS.md      # performance backlog
```

Header convention: each folder has a `main.hpp` aggregator that `#include`s the public headers of that module (so callers do `#include "physics/buoyancy/..."` etc.). Files named `*_cpu.cpp` / `*_gpu.cu` are the two backends of the same function family.

---

## 5. How a simulation runs (execution flow)

Every app's `main` follows the same recipe (see `apps/dhvc/main_gpu.cu` for the GPU reference). Tracing it is the fastest way to understand the codebase:

**Setup (once):**
1. `MpiContext` (`parallel/mpi`) → `MPI_Init` (RAII). Parse the TOML (`config/`).
2. Build `MpiTopology` + `Subdomain` (3D Cartesian, per-rank ranges) and the stretched `Grid` (`grid/`). On GPU, bind one rank per node-local A100 and mirror grid metrics to device (`core/gpu/gpu_grid.cu`).
3. Load boundary conditions from TOML (`boundary/problem_loader`) → `BoundaryCondition`. Build the `TdmaRegistry` (`linear_solver/tdma`) — one solver per axis.
4. Allocate `CpuFields`/`GpuFields` for U,V,W,P,T (`core/variables`). Initialize: synthetic perturbation **or** cross-mesh warm-start (`post/io` + `post/cpu/interp_cpu.cpp`), then one pressure projection.
5. Build the `MomentumSystem`, `ScalarSystem`, `PressureSystem` (`core/system`), and (NOB) the `PropertyFields` (`physics/properties`).

**Time loop (per step):**
1. **CFL** → `dt` from the velocity field (`driver/cfl`).
2. **Properties** (NOB) — update μ,1/ρ,κ,… from T (`physics/properties`).
3. **Scalar/energy** — assemble + 3-sweep ADI solve for T, scatter increment (`equation/scalar`). TDMA via the registry; wall BC folded by `core/...modify_tdma_row`.
4. **Momentum** — assemble RHS (incl. **incremental** `−dt·∇P_acc` from `equation/pressure`), add **buoyancy** (`physics/buoyancy`, face-interpolated T), 3-sweep ADI, **block-couple** the increments, update U,V,W (`equation/momentum`).
5. **Pressure projection** — assemble `∇·U*`, solve Poisson (pencil FFT + TDMA), project velocity divergence-free, `P_acc += δp` (`equation/pressure`).
6. **Halo + BC** — exchange ghosts (`core/...halo`, CUDA-aware on GPU), re-apply boundary conditions, zero W at the wall face.
7. **Diagnostics/stats** — divergence + monitor line every `nmonitor` (`driver/monitor`, `post/diagnostics`); accumulate turbulence statistics after `t_stat_start` (`post/statistics`).
8. **Checkpoint** — write a restart at run end / schedule (`post/io`, `driver/restart`).

**Teardown:** write final stats + restart; `main` returns → `MpiContext` destructor calls `MPI_Finalize`.

---

## 6. Component reference — every directory & file

Each table lists the role of every significant file. `cpu/*.cpp` and `gpu/*.cu` are the two backends of the same family; only the role is given once.

### `src/common/` — project-wide primitives
| File | Role |
|---|---|
| `types.hpp` | `real_t` (double; single via `-DMPMSTD_SINGLE_PRECISION`) and integer/index aliases |
| `direction.hpp` | `Direction{X,Y,Z}`, `Side{Minus,Plus}`, `Component{U,V,W}` + name/int helpers |
| `macros.hpp` | `kHaloWidth = 1`, host/device qualifiers, `CUDA_CHECK`, likely/unlikely hints |
| `main.hpp` | aggregator |

### `src/core/` — central data abstractions
| File | Role |
|---|---|
| `domain.hpp` | **Domain** — context-view bundling Grid + Topology + Subdomain + TdmaRegistry; the single argument passed to equation/physics/post functions |
| `field_cpu.hpp` / `field_gpu.hpp` | **CpuField** (host `std::vector`) / **GpuField** (device `cudaMalloc`); identical layout; `to_device()`/`to_host()` |
| `variables.hpp` | `Var{U,V,W,P,T}`, `Const{nu,alpha_T}`, `FieldStore` → `CpuFields`/`GpuFields` keyed containers |
| `system.hpp` | `MomentumSystem` (dU/dV/dW increments + RHS + ADI workspace), `ScalarSystem`, `PressureSystem` (Poisson engine handle + `P_acc`) |
| `bands.hpp` | banded tridiagonal storage in PaScaL_TDMA `[n_row × n_sys]` layout |
| `grid.hpp`, `halo.hpp`, `boundary.hpp`, `boundary_ops.hpp`, `mpi_topology.hpp`, `config.hpp` | thin re-export headers aliasing the infra types into `core::` |
| `cpu/halo_cpu.cpp` | MPI_Sendrecv halo-plane exchange |
| `cpu/boundary_ops_cpu.cpp` | `apply_ghost`, `modify_tdma_row` (wall-BC matrix fold), `sync_field` |
| `gpu/field_gpu.cu` | GpuField cudaMalloc/cudaFree |
| `gpu/halo_gpu.cu` | CUDA-aware MPI halo (per-axis pack/unpack; X contiguous, Y kernel-pack, Z memcpy3D) |
| `gpu/boundary_ops_gpu.cu` | GPU `apply_ghost` / `sync_field` |
| `gpu/gpu_grid.{hpp,cu}` | device copies of grid metric arrays (dx, dmx per axis) |

### `src/grid/` — stretched MAC grid
| File | Role |
|---|---|
| `grid.{hpp,cpp}` | **Grid** — builds **global** faces then slices this rank's band by `global_offset` (coords are global-physical → required for warm-start); accessors `xc/xf/dx/dmx(Direction)` |
| `stretching.{hpp,cpp}` | `make_face_coordinates(Uniform\|Tanh, n, L, gamma)` |
| `grid_device.{hpp,cpp}` | host→device mirror of grid arrays |

### `src/parallel/` — distributed-memory + device plumbing
| File | Role |
|---|---|
| `mpi/mpi_context.{hpp,cpp}` | **MpiContext** — RAII `MPI_Init`/`MPI_Finalize`; world/node rank, `node_comm` for GPU binding |
| `mpi/mpi_topology.{hpp,cpp}` | **MpiTopology** — 3D Cartesian comm + 3 axis comms + `comm_xz` pencil comm for the pressure FFT |
| `mpi/subdomain.{hpp,cpp}` | **Subdomain** — per-rank `n_interior/n_total/global_offset` + MPI derived datatypes for halo planes |
| `mpi/cuda_aware_mpi.{hpp,cpp}` | runtime flag for CUDA-aware MPI availability |
| `cuda/cuda_memory,cuda_stream,cuda_runtime,cuda_launch,error_check,nvtx_range` | thin cudaMalloc/stream/launch/error/profiling wrappers |
| `backend/{cpu,cuda}_backend.{hpp,cpp}`, `backend.hpp` | compile-time backend abstraction (alloc/free/synchronize) |

### `src/boundary/` & `src/config/`
| File | Role |
|---|---|
| `boundary/problem_loader.{hpp,cpp}` | parse TOML `[topology]` + `[bc.*.*]` → **Problem** |
| `boundary/problem.{hpp,cpp}` | Problem = per-axis/per-face boundary conditions |
| `boundary/face_bc.{hpp,cpp}`, `field_boundary.hpp`, `bc_kind.hpp` | per-face BC storage + kinds (Periodic/Dirichlet/Neumann/Wall) |
| `boundary/boundary_applier.{hpp,cpp}` | apply BCs to fields after a solve |
| `boundary/domain_topology.{hpp,cpp}` | MPI Cartesian topology + axis metadata helper |
| `config/config.{hpp,cpp}` | sectioned `key=value` loader, typed `get<T>()`/`get_or<T>()` |
| `config/logger.{hpp,cpp}` | rank-aware logging |

### `src/linear_solver/tdma/` — tridiagonal solvers
| File | Role |
|---|---|
| `tdma_solver.hpp` | abstract `TdmaSolver` (`solve_many`, cyclic variant); `[n_row × n_sys]` layout |
| `tdma_registry.{hpp,cpp}` | one solver per axis; `make_default()` picks CPU or CUDA backend at build time |
| `pascal_tdma_cpu_backend.{hpp,cpp}` | PaScaL_TDMA CPU (collective all-to-all distribution) |
| `pascal_tdma_cuda_backend.{hpp,cpp}` | PaScaL_TDMA CUDA (device-resident, CUDA-aware MPI) |
| `filtered_tdma_cpu_backend.hpp` | spectral-filter TDMA variant (channel) |

### `src/equation/` — the PDE solvers (each has `cpu/` + `gpu/`)
**momentum/**
| File | Role |
|---|---|
| `assemble.hpp`, `solve.hpp`, `kernels.hpp` | API: `assemble_momentum_{const,var}_visc_*`, `solve_momentum_*`, `update_velocity_*` |
| `cpu/momentum_cpu.cpp` | orchestrates assemble + 3-sweep ADI + block coupling |
| `cpu/momentum_var_kernels_cpu.cpp` | faithful NOB delta-form RHS + ADI bands + forward-sub coupling |
| `cpu/momentum_kernels_cpu.cpp` | OB (const-visc) stencils + `block_couple_dU/dV` |
| `cpu/momentum_cross_stress_cpu.cpp` | NOB transpose-stress terms (vanish at μ≡1) |
| `gpu/momentum_gpu.cu` | all CUDA kernels (`k_mom_rhs_var`, `k_mom_adi_var`, block-couple) + device buffer cache |

**scalar/** (temperature transport)
| File | Role |
|---|---|
| `assemble.hpp`, `solve.hpp`, `kernels.hpp`, `scalar_var_stencil.hpp` | API + NOB stencil helpers |
| `cpu/scalar_cpu.cpp` | assemble + 3-sweep ADI + update |
| `cpu/scalar_kernels_cpu.cpp` | OB explicit-Laplacian + convection stencils |
| `cpu/scalar_var_kernels_cpu.cpp` | faithful NOB var-κ energy residual + bands |
| `gpu/scalar_gpu.cu` | CUDA kernels + OB property cache |

**pressure/** (incremental van Kan projection)
| File | Role |
|---|---|
| `solve.hpp`, `assemble.hpp` | API: `solve_pressure_*`, `add_pressure_gradient_*` (predictor `−dt·∇P_acc`), `accumulate_pressure_*` (`P_acc += δp`) |
| `pressure_engine.hpp`, `pressure_base.hpp` | pencil-FFT engine + base interface |
| `cpu/pressure_cpu.cpp` | gradient/accumulate + lazy engine init |
| `cpu/pressure_engine_cpu.cpp` | full pencil pipeline: div RHS → FFT(x) → transpose → FFT(y) → TDMA(z) → inverse → project |
| `cpu/pressure_base_cpu.cpp` | ghost/halo helpers |
| `gpu/pressure_gpu.cu` | GPU gradient/accumulate kernels + `solve_pressure_gpu` |
| `gpu/pressure_engine_gpu.cuh` | GPU engine (cuFFT + device TDMA) |

### `src/solve/` — generic banded + Poisson orchestration
Earlier/general solver path (mixed FFT/DCT/TDMA per axis, banded Thomas spike). `poisson_solver.{hpp}` + `cpu/poisson_solver_cpu.cpp` + `gpu/poisson_solver_gpu.cu`; `banded_solver.{hpp}` + `cpu/banded_solver_cpu.cpp`. The production thermal-convection apps drive `equation/pressure` instead; `solve/` underpins the more generic cases.

### `src/physics/`
| File | Role | Status |
|---|---|---|
| `buoyancy/buoyancy.hpp` + `cpu/gpu` | OB buoyancy with **face-interpolated** T onto the gravity-axis velocity; NOB variant | active |
| `properties/properties.hpp` + `cpu/gpu` | `PropertyModel` (μ,ρ,κ,cp = polynomials of T; `glycerol(ΔT)` for NOB); unit constants at ΔT=0 | active |
| `forcing/forcing.hpp` + `cpu/gpu` | channel bulk-velocity / pressure-gradient forcing | active |
| `ibm/ibm.hpp` + `cpu/gpu` | immersed-boundary mask | **stub** |
| `les/les.hpp` + `cpu/gpu` | subgrid-scale viscosity | **stub** |

### `src/post/`
| File | Role |
|---|---|
| `statistics.{hpp}` + `cpu/statistics_cpu.cpp`, `gpu/statistics_gpu.cu` | Welford z-profile turbulence stats (global nx·ny norm), `compute_re_delta_star`, Nu |
| `io.{hpp}` + `cpu/io_cpu.cpp` | restart: `write_restart_gathered_cpu` (root-gather+fwrite), `read_restart_cpu` (MPI-IO), `read_global_field` (replicate), **`read_source_slab`** (Tier-1 distributed slab) |
| `cpu/interp_cpu.cpp` | cross-mesh warm-start: `make_src_mesh`, `compute_src_z_box`, `interp_global_to_field` (staggered trilinear) |
| `diagnostics.{hpp}` + `cpu/diagnostics_cpu.cpp`, `gpu/diagnostics_gpu.cu` | divergence / per-step diagnostic reductions |

### `src/driver/`
| File | Role |
|---|---|
| `cfl.{hpp}` + `cpu/cfl_cpu.cpp`, `gpu/cfl_gpu.cu` | adaptive `dt = min(MaxCFL/speed, dt_cap)` |
| `monitor.{hpp}` + `cpu/monitor_cpu.cpp` | per-step `step/t/dt/div/Re/KE/Nu/Trms` line + `monitor.csv` |
| `restart.{hpp}` + `cpu/restart_cpu.cpp` | checkpoint scheduling (`restart_due`) |

---

## 7. Applications (`apps/`)

| App | Entry files | Physics |
|---|---|---|
| **dhvc** | `main.cpp` (CPU), `main_gpu.cu` (full GPU incl. cuFFT pressure) | Differentially-heated vertical channel; x,y periodic, z wall (hot/cold). Primary case. |
| **rbc** | `main_{cpu,gpu}_{ob,nob}.{cpp,cu}` | Rayleigh–Bénard; z wall-normal (gravity). OB = constant property, NOB = glycerol variable-property. |
| **channel** | `main.cpp` (CPU) | Isothermal turbulent channel; bulk-velocity forcing. |
| **cavity** | `main.cpp` | Lid-driven cavity (all-Neumann pressure); framework. |
| **validate** | `main.cpp` | Fixed-IC, fixed-dt harness for byte-level comparison vs PaScaL_TCS. |

**TOML config sections** (`apps/dhvc/input_fig7_gpu_*.toml`):
```toml
[domain]  Lx Ly Lz            [mesh]  n1m n2m n3m   # n3m = wall-normal
[mpi]     np1 np2 np3          # GPU: 1 1 N
[uniform_mesh] uniform1/2/3    [mesh_stretch] gamma1/2/3   # z tanh γ=2.6
[physics] Ra Pr                [topology] x y = "periodic"  z = "wall"
[bc.z.minus] T=0.5             [bc.z.plus] T=-0.5
[time] t_start t_end dt_start MaxCFL n_steps
[init] perturb
[output] dir_stats nmonitor t_stat_start nstat t_stat_out
[restart] dir_in dir_out warmstart_io="slab"|"replicate" warmstart_dryrun
```
**Plotting:** `apps/dhvc/plot_dhvc_stats.ipynb` (time series), `apps/dhvc/plot_restart.ipynb` (U/V/W/T/P field slices), `apps/rbc/plot_rbc_fig9.py`.

---

## 8. Building

**Dependencies:** PaScaL_TDMA_C (`-lpascal_tdma`), FFTW3 (host Poisson, used even in GPU build), CUDA/cuFFT via **NVHPC 23.7** (`nvcc 12.2`, `-cudalib=cufft`), MPI (NVHPC mpic++ / system mpicxx).

```bash
# CPU (login node, CUDA-free)
make BACKEND=cpu all            # → build/cpu/{lib/libmpmstd.a, bin/apps/*}

# GPU — MUST run on a GPU node (nvc++ needs the driver for .cu)
module load nvhpc/23.7
make BACKEND=cuda all           # → build/cuda/bin/apps/{dhvc_gpu, rbc_gpu_ob/nob, …}

make BACKEND=cuda lib           # library only
make clean                      # wipe build/$(BACKEND)
make info                       # show resolved build variables
```
GPU SLURM scripts compile `dhvc_gpu` **directly** (bypassing the make target to dodge NFS Makefile-cache lag) — see `scripts/dhvc_compile.sh`:
```
mpic++ -std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA \
       $INCS -c apps/dhvc/main_gpu.cu -o dhvc_gpu.o
mpic++ ... dhvc_gpu.o build/cuda/lib/libmpmstd.a $EXT -o build/cuda/bin/apps/dhvc_gpu
```
Targets: `channel cavity dhvc rbc validate rbc_cpu_ob rbc_cpu_nob` (+ `rbc_gpu_ob rbc_gpu_nob dhvc_gpu` under cuda).

---

## 9. Running

Binaries take one TOML argument. Run jobs **visible** (`srun` / `sbatch --wait`), not background-polled.
```bash
# CPU channel smoke
mpirun -np 4 build/cpu/bin/apps/channel apps/channel/input_smoke.toml
# GPU DHVC (2×A100) via SLURM
sbatch scripts/dhvc_gpu_1e5.sh   # mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e5.toml
```
**SLURM scripts** (`scripts/`): `dhvc_gpu_1e5/1e6.sh` (2×A100, np=2), `dhvc_gpu_1e7/1e8.sh` (4×A100/2 nodes, np=4), `dhvc_compile.sh` (build), `tier1_validate.sh` (build + warm-start A/B + auto-launch next rung), RBC Fig9 scripts. Pattern: `--gres=gpu:a100:N`, `--nodes`, `mpirun -np <np1*np2*np3>`.

---

## 10. Restart & cross-mesh warm-start

Restart = one **decomposition-independent** global C-order `.bin` per field (`U/V/W/T/P.bin`) + `meta.txt` (step, time, n1m/n2m/n3m, Ra — **no rank info**). So you can restart with a **different GPU count** or a **different mesh**.

- **Same-mesh continuation** — `read_restart_cpu` (each rank reads its subdomain by `global_offset`).
- **Cross-mesh warm-start** (Ra-ladder 1e6→1e7→1e8…) — coordinate-based **staggered trilinear interpolation** from a developed coarser field, then **one pressure projection** for ∇·u=0. Each variable interpolated at its own MAC location; x,y periodic-wrap, z wall-clamp. Avoids cold-start conduction-shock dt collapse.

**Warm-start I/O** (`[restart] warmstart_io`):
- `replicate` — root reads whole field + `MPI_Bcast` (per-rank memory = full field; doesn't scale).
- **`slab` (default)** — each rank reads only its source z-slab via collective `MPI_File_read_all` + subarray (per-rank memory ∝ `full_field/N_z`, parallel I/O). Same trilinear math → bit-for-bit identical to `replicate` (the A/B test in `tier1_validate.sh`). `warmstart_dryrun=1` dumps interpolated-field checksums and exits.

---

## 11. Tests, reports, docs

- **`tests/unit/`** — field layout, grid/stretching, halo, BC application, TDMA backend, stencils, sweep order.
- **`tests/integration/`** — momentum/thermal advection order, Dirichlet wall, manufactured heat eqn, Boussinesq buoyancy, pressure-Poisson (serial/DCT/MPI), restart round-trip.
- **`tests/regression/`** + **`apps/validate`** — field-to-field comparison vs a PaScaL_TCS ground truth (the correctness anchor).
- Build & run: `make BACKEND=cpu tests` → `mpirun -np 2 build/cpu/bin/tests/unit/test_grid_multi`.

- **`Report/01…20+`** — chronological design + status: architecture decisions, momentum-fidelity audits vs PaScaL_TCS, GPU/CPU bottleneck analyses, OB-vs-NOB scheme audits, incremental-pressure audit.
- **`docs/`** — `NOB_RBC_SPEC.md`, `PROGRESS.md`, `REFACTOR_PLAN.md` (dual-build rev.2), `STRUCTURE_REDESIGN.md`.
- **`OPTIMIZATIONS.md`** — perf backlog. Known wall: explicit convection → CFL dt collapse at high Ra; durable fix = fully-implicit convection. GPU healthy regime ~50 ms/step (1e6, 2×A100) → ~315 ms/step (1e7, 4×A100).

---

*For ground truth, read the code. `Report/` and the project memory notes capture the "why" behind non-obvious choices (axis swap, wall-BC matrix fold, free-fall scaling, Tier-1 slab I/O).*
