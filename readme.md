# MPM-STD_C

**MPM-STD_C is my own multi-GPU DNS solver** for **wall-bounded thermal convection** — a unified **C++/CUDA** incompressible-flow code I designed and implemented from the ground up. One source tree produces both **CPU (MPI)** and **GPU (CUDA + MPI)** binaries through a folder-selected "dual build", and the GPU path scales across multiple A100s via z-split CUDA-aware MPI.

It solves two canonical DNS benchmarks — **DHVC** (Differentially-Heated Vertical Channel) and **RBC** (Rayleigh–Bénard Convection, both constant-property OB and variable-property NOB) — plus auxiliary **channel**, **cavity**, and a **validate** harness.

> Numerics are validated against the Fortran reference solver **PaScaL_TCS** and published DNS (Ng et al., Kim et al. 2023). PaScaL_TCS is only the correctness yardstick; all code here is my own implementation.

This README is a **project map** kept in sync with the current tree: read it top-to-bottom to understand how the codebase is structured, how a timestep flows through the modules, and what every directory does.

---

## Table of contents
1. [The big picture](#1-the-big-picture)
2. [Physics & numerical method](#2-physics--numerical-method)
3. [Conventions you must know (gotchas)](#3-conventions-you-must-know-gotchas)
4. [Project structure at a glance](#4-project-structure-at-a-glance)
5. [How a simulation runs (execution flow)](#5-how-a-simulation-runs-execution-flow)
6. [Component reference — every directory](#6-component-reference--every-directory)
7. [Applications (`apps/`)](#7-applications-apps)
8. [Building](#8-building)
9. [Running](#9-running)
10. [Restart & cross-mesh warm-start](#10-restart--cross-mesh-warm-start)
11. [Tests, reports, docs](#11-tests-reports-docs)

---

## 1. The big picture

The code is **one static library** (`libmpmstd.a`, built from `src/`) plus thin **application drivers** (`apps/*/main*.{cpp,cu}`). Each app wires the library modules into an explicit time loop for one physics case.

The library is a stack of **numeric layers** on top of **host infrastructure**:

```
            apps/ (dhvc, rbc, channel, cavity, validate)  ← thin drivers: config → assemble → time loop
            ───────────────────────────────────────────
 numeric    driver/      cfl (adaptive dt)
 layers     post/        restart I/O, cross-mesh warm-start, per-case statistics objects
            physics/     buoyancy, properties (NOB μ,ρ,κ,cp), forcing
            equation/    momentum · scalar(energy) · pressure   ← the PDE solvers
            core/        Domain, Field (cpu/gpu), FieldStore, halo, BC ops, device grid
            ───────────────────────────────────────────
 host       grid/        stretched MAC grid (xc/xf/dx/dmx)
 infra      boundary/    BoundaryCondition parsed from the input file
            linear_solver/  PaScaL_TDMA registry (one concrete backend per build)
            parallel/    MPI topology/subdomain + CUDA helpers
            config/      key=value loader + logger
            common/      real_t, Direction enum, macros (kHaloWidth)
```

**Dual build.** The backend is selected by **folder**: numeric layers have a `cpu/` (host `.cpp`) and a `gpu/` (`.cu`) subfolder implementing the same function family with `_cpu` / `_gpu` suffixes; logic shared by both lives in the module's top-level header. `make BACKEND=cpu` compiles `cpu/` + backend-agnostic code; `make BACKEND=cuda` additionally compiles the `gpu/` `.cu` (with the same NVHPC compiler). The caller picks the backend by calling `foo_cpu(...)` vs `foo_gpu(...)`.

**Namespace = location.** A type at `mpmstd::a::b::Name` is defined under `src/a/b/`; there are no re-export aliases (call sites use the real namespace, e.g. `parallel::mpi::Subdomain`, `boundary::BoundaryCondition`, `equation::momentum::…`). `cpu/`·`gpu/` folders are an implementation detail, not a namespace level. A lint script enforces this (`make lint`).

---

## 2. Physics & numerical method

**Equations.** Incompressible Navier–Stokes + energy with Boussinesq buoyancy (OB) or full variable-property coupling (NOB).

**Non-dimensionalization — free-fall scaling.** Velocity = free-fall velocity `U_ff = √(gβΔT H)`, length = channel width `H` → **1 time unit = 1 free-fall time** `H/U_ff`. Hence `nu = √(Pr/Ra)`, `alpha = 1/√(Ra·Pr)`. Every monitor/stats quantity (`t, dt, Re, KE, Nu, Trms, …`) is in these units.

**Per-timestep scheme.**
- **Momentum** — Beam–Warming **delta-form ADI** (3 sweeps): own-direction diffusion implicit ×2, cross ×1; **semi-implicit convection**. After the sweeps, **block lower-triangular velocity coupling** corrects the increments (cross-velocity convection of δU/δV/δW).
- **Scalar (temperature)** — ADI; OB = explicit convection + Crank–Nicolson diffusion; NOB = semi-implicit variable-κ kernels.
- **Pressure** — **incremental (van Kan) projection**: the predictor subtracts `dt·∇P_acc`; a Poisson solve gives the increment `δp`; velocity is projected divergence-free; `P_acc += δp`. The Poisson solve = **pencil FFT in (x,y) + distributed TDMA in z** (FFTW / cuFFT + PaScaL_TDMA). `P_acc` lives in `Var::P`; the per-step increment `δp` is owned by the pressure `LinearSystem`.
- **Constant- vs variable-property** — the `*_variable_property` kernels are the faithful scheme; at `ΔT=0` the property model → unit constants and they reduce **exactly** to the constant-property (OB) form. DHVC runs the `*_variable_property` kernels at unit properties; dedicated `*_constant_property` kernels also exist and are live for `rbc_gpu_ob` / the CPU OB path.

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

- **`z` is the wall-normal axis** (PaScaL_TCS uses `y`). Map the paper's `Nx×Ny×Nz` (Ny=wall, Nz=span) → our `(n1m, n2m, n3m) = (Nx, Nz_span, Ny_wall)`. Swapping wrong under-resolves the boundary layer.
- **GPU runs z-split only** (`np1=1, np2=1, np3=N`) — the fast path (CUDA-aware MPI halos, no whole-field host staging). x/y split forces slow host-staged FFT transposes.
- **MPI-only parallelism** — no OpenMP / std::thread / FFTW threads.
- **Wall BC is folded into the TDMA matrix** (zero-ghost for cell-centered U/V, not antisymmetric); the W face at the z-wall (interior index `k=h`) is zeroed explicitly after the W solve.
- **Full identifiers, no abbreviations** (`domain` not `dom`, `momentum`/`pressure` not `mom`/`poi`).
- **Host vs device locals carry a prefix** when both live in one translation unit: `cpu_fields` / `gpu_fields`, `cpu_properties` / `gpu_properties`.
- **Restarts are decomposition-independent** — one global C-order `.bin` per field; restart with a different GPU count or mesh (§10).
- **Adaptive dt is `min(dt_cfl, dt_cap)`** — never `max(...)` (that pins dt and blows up explicit convection).
- **Config files use a `.toml` extension but the parser is INI-style** (`config/config.hpp`, "Minimal INI-style"): flat `[section]` headers + `key = value`. `[bc.z.minus]` works because it is treated as one flat section name; genuine TOML arrays / inline tables are **not** parsed. Unknown/mistyped optional keys fall through to a default, but each binary **echoes its parsed config and warns about keys it never read** at startup (`Config::echo` / `warn_unused_keys`) so typos surface. Full key table: [docs/config_reference.md](docs/config_reference.md).

---

## 4. Project structure at a glance

```
MPM-STD_C/
├── src/                  # the library (libmpmstd.a)
│   ├── common/           # real_t, Direction, kHaloWidth, host/device macros
│   ├── core/             # Domain, Field(cpu/gpu), FieldStore(fields.hpp), halo, BC ops, device grid
│   ├── grid/             # stretched MAC grid (xc/xf/dx/dmx) + device copy
│   ├── parallel/         # mpi/ (context/topology/subdomain) + cuda_helpers/
│   ├── boundary/         # BoundaryCondition + face BCs, loaded from the input file
│   ├── config/           # INI-style config loader + logger
│   ├── linear_solver/    # tdma/ (PaScaL_TDMA registry; one concrete backend per build)
│   ├── equation/         # momentum/ scalar/ pressure/   ← the PDE solvers (each has cpu/ + gpu/)
│   ├── physics/          # buoyancy/ properties/ forcing/ (each has cpu/ + gpu/)
│   ├── post/             # restart, warm-start interp, per-case statistics objects
│   └── driver/           # cfl (adaptive dt)
├── apps/                 # entry points + input files + plotting scripts/notebooks
│   ├── dhvc/  rbc/  channel/  cavity/  validate/
├── scripts/              # SLURM batch scripts (build, run, experiment ladders) + lint
├── tests/                # unit / integration / regression (built by `make tests`)
├── test/                 # older orphan unit tests (not yet wired into the build)
├── Report/               # design + status reports; Report/convenience/ = the de-abstraction plans
├── docs/                 # specs + conventions (NOB spec, namespace convention, history)
├── Makefile  Makefile.inc   # dual (cpu/cuda) build
└── OPTIMIZATIONS.md      # performance backlog
```

There is **no** umbrella `main.hpp` per folder anymore — callers `#include` the specific header they need (`equation/momentum/solve.hpp`, `post/restart.hpp`, …). Files named `*_cpu.cpp` / `*_gpu.cu` are the two backends of one function family.

---

## 5. How a simulation runs (execution flow)

Every app's `main` follows the same recipe (`apps/rbc/main_gpu_nob.cu` is the most complete GPU reference). Tracing it is the fastest way to understand the codebase.

**Setup (once):**
1. `MpiContext` (`parallel/mpi`) → `MPI_Init` (RAII). Load the config (`config/`); read every input value up front.
2. Build `MpiTopology` + `Subdomain` (3D Cartesian, per-rank ranges) and the stretched `Grid` (`grid/`). On GPU, bind one rank per node-local A100 and mirror grid metrics to device (`core/gpu/gpu_grid.cu`).
3. Load boundary conditions (`boundary/problem_loader`) → `BoundaryCondition`. Build the `TdmaRegistry` (`linear_solver/tdma`) — one solver per axis.
4. Allocate `CpuFields` / `GpuFields` for U,V,W,P,T (`core/fields.hpp`). Initialize: synthetic perturbation (cold start) **or** cross-mesh warm-start (`post::DhvcRestart` / `post::Restart`), then upload to device.
5. Build the per-equation workspaces `equation::{momentum,scalar}::GpuLinearSystem` and `equation::pressure::LinearSystem`, plus (NOB) `physics::properties::GpuPropertyFields` and the `physics::buoyancy::BuoyancyParams`, and the statistics object (`post::RbcStatistics` / `post::DhvcStatistics`).

**Time loop (per step):**
1. **CFL** → `dt` from the velocity field (`driver/cfl`).
2. **Properties** (NOB) — update μ,1/ρ,κ,… from T (`physics/properties`).
3. **Scalar/energy** — assemble + 3-sweep ADI solve for T (`equation/scalar`); TDMA via the registry, wall BC folded in the matrix.
4. **Momentum** — assemble RHS, add the **incremental** predictor `−dt·∇P_acc` (`equation/pressure`) and **buoyancy** (`physics/buoyancy`, face-interpolated T), 3-sweep ADI, **block-couple** the increments, update U,V,W (`equation/momentum`).
5. **Wall W** — zero the W face at the z-wall.
6. **Pressure projection** — solve Poisson (pencil FFT + TDMA), project velocity divergence-free, `P_acc += δp` (`equation/pressure`).
7. **Statistics / monitor** — device-side reductions each step; a `monitor.csv` line every `nmonitor`; a cumulative-average profile every `t_stat_out` (`post::{Rbc,Dhvc}Statistics`).
8. **Checkpoint** — periodic + final restart (`post::{Restart,DhvcRestart}`), staging device→host via `core::copy_all_to_host`.

**Teardown:** finalize stats + write restart; `main` returns → `MpiContext` destructor calls `MPI_Finalize`.

---

## 6. Component reference — every directory

`cpu/*.cpp` and `gpu/*.cu` are the two backends of one family; the role is given once.

### `src/common/` — project-wide primitives
`types.hpp` (`real_t` = double; `-DMPMSTD_SINGLE_PRECISION` for float), `direction.hpp` (`Direction{X,Y,Z}`, `Side{Minus,Plus}` + helpers), `macros.hpp` (`kHaloWidth = 1`, `MPMSTD_HD` host/device qualifier, `CUDA_CHECK`).

### `src/core/` — central data abstractions
| File | Role |
|---|---|
| `domain.hpp` | **Domain** — bundles Grid + Topology + Subdomain + TdmaRegistry; the single argument passed to equation/physics/post functions |
| `field_cpu.hpp` / `field_gpu.hpp` | **CpuField** (host `std::vector`) / **GpuField** (device `cudaMalloc`); identical layout; `to_device()` / `to_host()` |
| `fields.hpp` | `Var{U,V,W,P,T}`, `Const{nu,alpha_T}`, `FieldStore` → `CpuFields` / `GpuFields` keyed containers; `core::copy_all_to_host(GpuFields, CpuFields)` |
| `gpu_binding.hpp` | `parallel::mpi::Subdomain` + GPU-rank binding helpers |
| `boundary_ops.hpp` | `apply_ghost`, `modify_tdma_row` (wall-BC matrix fold), `sync_field` |
| `halo.hpp` | halo-exchange API |
| `grid.hpp`, `boundary.hpp`, `config.hpp` | thin include-forwarders to the real `grid/`, `boundary/`, `config/` headers |
| `cpu/halo_cpu.cpp`, `cpu/boundary_ops_cpu.cpp` | host halo + BC ops |
| `gpu/field_gpu.cu`, `gpu/halo_gpu.cu`, `gpu/boundary_ops_gpu.cu` | device field alloc, CUDA-aware MPI halo, GPU BC ops |
| `gpu/gpu_grid.{hpp,cu}` | device copies of grid metric arrays (dx, dmx per axis) |

### `src/grid/` — stretched MAC grid
`grid.{hpp,cpp}` builds **global** faces then slices this rank's band by `global_offset` (coords are global-physical → required for warm-start); `stretching.{hpp,cpp}` = `make_face_coordinates(Uniform | Tanh, n, L, γ)`; `grid_device.{hpp,cpp}` = host→device mirror.

### `src/parallel/` — distributed-memory + device plumbing
- `mpi/mpi_context` — RAII `MPI_Init`/`Finalize`; world/node rank, `node_comm` for GPU binding.
- `mpi/mpi_topology` — 3D Cartesian comm + 3 axis comms + `comm_xz` pencil comm for the pressure FFT.
- `mpi/subdomain` — per-rank `n_interior/n_total/global_offset` + MPI derived datatypes for halo planes.
- `mpi/cuda_aware_mpi` — runtime flag for CUDA-aware MPI.
- `cuda_helpers/` — thin cudaMalloc / stream / launch / error-check / NVTX wrappers.

### `src/boundary/` & `src/config/`
- `boundary/problem_loader` — parse `[topology]` + `[bc.*.*]` → **BoundaryCondition**.
- `boundary/problem` — `BoundaryCondition` = per-axis / per-face conditions; `face_bc`, `field_boundary`, `bc_kind` (Periodic / Dirichlet / Neumann); `domain_topology` = Cartesian topology helper.
- `config/config` — INI-style `[section]` `key=value` loader, typed `get<T>()` / `get_or<T>()` (throws on missing required key); `config/logger` — rank-aware logging.

### `src/linear_solver/tdma/` — tridiagonal solvers
`tdma_registry.{hpp,cpp}` holds one solver per axis and defines `using TdmaBackend = PaScaLTDMACpuBackend | PaScaLTDMACudaBackend` — the **one concrete backend for this build** (chosen by `#ifdef MPMSTD_BACKEND_CUDA`; no virtual base). `pascal_tdma_cpu_backend` = PaScaL_TDMA CPU (collective all-to-all); `pascal_tdma_cuda_backend` = PaScaL_TDMA CUDA (device-resident, exposes `solve_many_device`).

### `src/equation/` — the PDE solvers (each: `cpu/` + `gpu/`)
Namespaces `equation::{momentum,scalar,pressure}`. The API verb is the numerical fact — `_constant_property` (OB) / `_variable_property` (NOB) — not the physics model.

- **momentum/** — `assemble.hpp` (`assemble_momentum_{constant,variable}_property_*`), `solve.hpp` (`solve_momentum_*`, `update_velocity_*`), `kernels.hpp`, `linear_system.hpp` (`{Cpu,Gpu}LinearSystem` = δU/δV/δW increments + RHS + ADI workspace + block coupling). Impl: `cpu/momentum_*_cpu.cpp`, `gpu/momentum_gpu.cu`.
- **scalar/** (energy) — `assemble.hpp`, `solve.hpp` (`solve_scalar_{constant,variable}_property_*`), `kernels.hpp`, `scalar_var_stencil.hpp`, `linear_system.hpp`. Impl: `cpu/scalar_*_cpu.cpp`, `gpu/scalar_gpu.cu`.
- **pressure/** (incremental van Kan) — `assemble.hpp` (`add_pressure_gradient_*` predictor, `accumulate_pressure_*`), `solve.hpp` (`solve_pressure_{constant,variable}_property_*`), `pressure_engine.hpp` (the concrete pencil-FFT `PressureSolver` — the former abstract base is merged in), `linear_system.hpp` (`LinearSystem` owns the `PressureSolver` + the `δp` increment). Impl: `cpu/pressure_engine_cpu.cpp`, `cpu/pressure_base_cpu.cpp` (field-agnostic pencil steps), `gpu/pressure_engine_gpu.cuh`, `gpu/pressure_gpu.cu`.

### `src/physics/` (each: `.hpp` + `cpu/` + `gpu/`)
- `buoyancy/` — OB buoyancy with **face-interpolated** T onto the gravity-axis velocity + NOB variant; `BuoyancyParams::from_property_model(model, gravity_axis)` derives the NOB density-nonlinearity params.
- `properties/` — `PropertyModel` (μ,ρ,κ,cp = polynomials of T; `physics::properties::glycerol(ΔT)` for NOB); unit constants at ΔT=0. `GpuPropertyFields` holds the per-cell effective coefficients.
- `forcing/` — channel bulk-velocity / pressure-gradient forcing.

### `src/post/`
| File / object | Role |
|---|---|
| `io.{hpp}` + `cpu/io_cpu.cpp` | low-level field I/O: `write_restart_gathered_cpu` (root-gather + fwrite — NFS-safe), `read_restart_cpu`, `read_global_field` (replicate), **`read_source_slab`** (Tier-1 distributed slab), plus the warm-start mesh helpers |
| `cpu/interp_cpu.cpp` | cross-mesh warm-start math: `make_src_mesh`, `compute_src_z_box`, `interp_global_to_field` (staggered trilinear) |
| `restart.hpp` + `cpu/restart_cpu.cpp` | **`post::Restart`** (RBC) — `load` / `save` / `checkpoint_if_due`; generic `RestartState{step, time, extras[]}` so the stats payload is byte-compatible |
| `dhvc_restart.hpp` + `cpu/dhvc_restart_cpu.cpp` | **`post::DhvcRestart`** — cross-mesh warm-start `load_warmstart()` (interpolate any source mesh → this grid) + `save()` checkpoint with `meta.txt` |
| `statistics.{hpp}` + `cpu/statistics_cpu.cpp`, `gpu/statistics_gpu.cu` | `post::Stats` z-profile turbulence stats, `compute_re_delta_star`, Tecplot writer |
| `statistics_rbc.hpp` + `gpu/statistics_rbc_gpu.cu` | **`post::RbcStatistics`** — device Θc / Nu / z-profile reductions + `monitor.csv` + `stats_t*.dat` + restart bridging |
| `statistics_dhvc.hpp` + `gpu/statistics_dhvc_gpu.cu` | **`post::DhvcStatistics`** — device Re_δ* / KE / Nu / z-profile reductions + `monitor.csv` + stats file-out |

### `src/driver/`
`cfl.{hpp}` + `cpu/cfl_cpu.cpp`, `gpu/cfl_gpu.cu` — adaptive `dt = min(MaxCFL / speed, dt_cap)`.

---

## 7. Applications (`apps/`)

| App | Entry files | Physics |
|---|---|---|
| **dhvc** | `main.cpp` (CPU), `main_gpu.cu` (full GPU incl. cuFFT pressure) | Differentially-heated vertical channel; x,y periodic, z wall (hot/cold), gravity on the streamwise component. Primary case. |
| **rbc** | `main.cpp` (monolith), `main_cpu_ob.cpp`, `main_cpu_nob.cpp`, `main_gpu_ob.cu`, `main_gpu_nob.cu`, shared `rbc_common.hpp` | Rayleigh–Bénard; z wall-normal (gravity). OB = constant-property, NOB = glycerol variable-property. `main_gpu_nob.cu` is the production binary. |
| **channel** | `main.cpp` (CPU) | Isothermal turbulent channel; bulk-velocity forcing. |
| **cavity** | `main.cpp` | Lid-driven cavity (all-Neumann pressure). |
| **validate** | `main.cpp` | Fixed-IC, fixed-dt serial harness — dumps fields for byte-level regression comparison. |

**Config sections** (INI-style; full table with types, defaults, and which binary reads each key in [docs/config_reference.md](docs/config_reference.md)):
```
[mpi]           np1 np2 np3                         # GPU: 1 1 N
[topology]      x  y  z                             # "periodic" | "wall"
[mesh]          n1m n2m n3m                         # n3m = wall-normal
[domain]        Lx Ly Lz
[uniform_mesh]  uniform1 uniform2 uniform3          # 1 = uniform, 0 = stretched
[mesh_stretch]  gamma1 gamma2 gamma3                # tanh γ (z ≈ 2.6)
[physics]       Ra Pr   [DeltaT gravity_axis]       # DeltaT/gravity_axis: RBC NOB
[time]          t_start t_end dt_start MaxCFL n_steps
[init]          perturb
[output]        dir_stats nmonitor t_stat_start t_stat_out
[restart]       dir_in dir_out                      # both empty = cold start, no checkpoint
                nrestart_write stats_reset          # RBC: periodic ckpt + fresh-stats flag
                warmstart_io warmstart_dryrun       # DHVC: "slab"(default)|"replicate"; dry-run
```
**Plotting:** `apps/dhvc/plot_dhvc_stats.ipynb`, `apps/dhvc/plot_restart.ipynb`, `apps/dhvc/plot_dhvc_fig7.py`, `apps/rbc/` fig9 scripts.

---

## 8. Building

**Dependencies:** PaScaL_TDMA_C (`-lpascal_tdma`), FFTW3 (host Poisson — linked even in the GPU build), CUDA / cuFFT via **NVHPC 23.7** (`nvcc 12.2`, `-cudalib=cufft`), MPI (NVHPC `mpic++` for cuda, system `mpicxx` for cpu). Paths are set in `Makefile.inc`.

```bash
# CPU (login node, CUDA-free)
make BACKEND=cpu all            # → build/cpu/{lib/libmpmstd.a, bin/apps/*}
#   equivalently: make cpu

# GPU — MUST run on a GPU node (nvc++ needs the driver to compile .cu)
module load nvhpc/23.7
make BACKEND=cuda all           # → build/cuda/bin/apps/{dhvc_gpu, rbc_gpu_{ob,nob}, …}

make BACKEND=<b> lib            # library only
make BACKEND=<b> tests          # build tests/ into build/<b>/bin/tests/
make lint                       # namespace=location convention check
make clean                      # wipe build/$(BACKEND)
make info                       # print resolved build variables
```

App targets — cpu: `channel cavity dhvc rbc validate rbc_cpu_ob rbc_cpu_nob`; cuda adds `rbc_gpu_ob rbc_gpu_nob dhvc_gpu`.

> ⚠️ **Do not `source` the Intel oneAPI env before a CPU build** — its `libmpi.so.12` clashes with the OpenMPI (`libmpi.so.40`) that FFTW/PaScaL_TDMA link. Use the default `/usr/bin/mpicxx`.

GPU SLURM scripts under `scripts/` compile the `.cu` mains directly (dodging NFS Makefile-cache lag) — e.g. `scripts/dhvc_compile.sh`, `scripts/compile_rbc_gpu_nob.sh`.

---

## 9. Running

Each binary takes one input file. Run jobs **visibly** (`srun` / `sbatch --wait`), not background-polled.
```bash
# CPU RBC (NOB) smoke
mpirun -np 2 build/cpu/bin/apps/rbc_cpu_nob apps/rbc/input_reorder_cold_dt40.toml
# GPU DHVC (2×A100) via SLURM
sbatch scripts/dhvc_gpu_1e5.sh   # → mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e5.toml
```
`scripts/` holds the batch scripts: `dhvc_gpu_1e5/1e6/1e7/1e8.sh`, RBC Fig9 (`g25*.sh`) and the reorder/deltat ladders, plus `tier1_validate.sh` (build + warm-start slab-vs-replicate A/B). Pattern: `--gres=gpu:a100:N`, `mpirun -np <np1·np2·np3>`.

> ⚠️ On GPU, put input/output paths on **shared NFS** (`/shared/...`) — `/tmp` is node-local and invisible to the compute node.

---

## 10. Restart & cross-mesh warm-start

A restart is one **decomposition-independent** global C-order `.bin` per field (`U/V/W/T/P.bin`, `P` = accumulated pressure) + a metadata file — **no rank info** — so you can restart with a **different GPU count** or a **different mesh**.

- **RBC** (`post::Restart`) — `stats.txt` metadata (step, time + running stats). Same-mesh continuation; `checkpoint_if_due` writes alternating `ckpt_a` / `ckpt_b` so a half-written checkpoint never clobbers the last good one. The load→save path is **byte-exact** (verified round-trip).
- **DHVC** (`post::DhvcRestart`) — `meta.txt` metadata (step, time, mesh dims, Ra). The load is a **cross-mesh warm-start**: read a developed coarser field (any mesh, discovered from `meta.txt`) and **staggered trilinear-interpolate** onto this grid (each variable at its own MAC location; x,y periodic-wrap, z wall-clamp), then **one pressure projection** for ∇·u=0. Avoids the cold-start conduction-shock dt collapse when climbing an Ra-ladder (1e6→1e7→1e8).

**Warm-start I/O** (`[restart] warmstart_io`):
- `replicate` — root reads the whole field + `MPI_Bcast` (per-rank memory = full field; doesn't scale).
- **`slab` (default)** — each rank reads only its source z-slab via collective `MPI_File_read_all` + subarray (per-rank memory ∝ `full_field/N_z`). Same trilinear math → bit-for-bit identical to `replicate`. `warmstart_dryrun=1` dumps interpolated-field checksums and exits.

Both cases stage device→host with `core::copy_all_to_host` before the gathered write.

---

## 11. Tests, reports, docs

- **`tests/`** (built by `make BACKEND=cpu tests`) — unit + integration + regression checks; the `validate` app dumps fields for byte-level comparison against a PaScaL_TCS ground truth.
- **`test/`** — an older orphan set of unit tests, not yet wired into the Makefile (a stage-4 cleanup item).
- **`Report/`** — chronological design + status reports (architecture, momentum-fidelity audits vs PaScaL_TCS, GPU/CPU bottleneck analyses, OB-vs-NOB and incremental-pressure audits). `Report/convenience/` holds the de-abstraction / main-cleanup plans (`01`–`03`).
- **`docs/`** — [`config_reference.md`](docs/config_reference.md) (every input key: type, default, which binary reads it), [`output_formats.md`](docs/output_formats.md) (monitor.csv / stats / restart file layouts), `NOB_RBC_SPEC.md`, `namespace_convention.md` (the namespace=location rule + `make lint`), `PROGRESS.md`, `REFACTOR_PLAN.md`, `STRUCTURE_REDESIGN.md`.
- **`OPTIMIZATIONS.md`** — perf backlog. Known wall: explicit convection → CFL dt collapse at high Ra; the durable fix is fully-implicit convection.

---

*For ground truth, read the code. `Report/` and the project memory notes capture the "why" behind non-obvious choices (axis swap, wall-BC matrix fold, free-fall scaling, Tier-1 slab I/O, the incremental-pressure `P_acc`/`δp` split, and the `src/post/` statistics-object extraction).*
