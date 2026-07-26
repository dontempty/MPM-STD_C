# Config key reference

Input files (`apps/*/*.toml`) are read by an **INI-style parser** (`src/config/config.hpp`,
"Minimal INI-style") — flat `[section]` headers + `key = value`. The `.toml`
extension is historical: genuine TOML arrays / inline tables are **not** parsed,
and `[bc.z.minus]` works only because it is one flat section name.

- **Required** keys are read with `get<T>()` → the run **throws** if they are missing.
- **Optional** keys are read with `get_or<T>(…, default)` → a missing/mistyped key
  **silently takes the default**.
- Every binary echoes its parsed config at startup and prints
  `[config] WARNING: … key(s) … were never read (typo…?)` for keys in the file it
  never used (see `Config::echo` / `warn_unused_keys`). `np1*np2*np3` must equal
  `mpirun -np` or `MpiTopology` throws.

Defaults **differ per binary** — the table gives RBC (`rbc_gpu_nob`) and DHVC
(`dhvc_gpu`), the two production cases. Channel/cavity/validate share the same
loader with their own defaults.

## `[mpi]` — decomposition (required)
| Key | Type | Meaning |
|---|---|---|
| `np1` `np2` `np3` | int | ranks per axis; product must equal `mpirun -np`. **GPU: `1 1 N`** (z-split only). |

## `[topology]` — periodicity (optional, default shown)
| Key | Type | Default | Values |
|---|---|---|---|
| `x` `y` `z` | string | `periodic` `periodic` `wall` | `"periodic"` \| `"wall"` |

## `[mesh]` / `[domain]` — grid (required)
| Key | Type | Meaning |
|---|---|---|
| `n1m` `n2m` `n3m` | int | global cell counts; **`n3m` = wall-normal (z)** |
| `Lx` `Ly` `Lz` | double | domain lengths |

## `[uniform_mesh]` / `[mesh_stretch]` — stretching (optional)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `uniform1` `uniform2` `uniform3` | int | `1` | `1` = uniform axis, `0` = tanh-stretched |
| `gamma1` `gamma2` `gamma3` | double | `0.0` | tanh stretching factor (z typically `2.6`) |

## `[physics]`
| Key | Type | Req/Default | Meaning |
|---|---|---|---|
| `Ra` | double | **required** | Rayleigh number |
| `Pr` | double | **required** | Prandtl number (RBC ≈ 2547 glycerol; DHVC 0.7) |
| `DeltaT` | double | `0.0` (RBC) | dimensional ΔT; `>0` activates NOB properties. Unused by DHVC. |
| `gravity_axis` | int | `2` (RBC) | gravity/buoyancy axis (RBC z=2). DHVC fixes it to 0 (streamwise) in code. |

## `[time]`
| Key | Type | Default | Meaning |
|---|---|---|---|
| `t_end` | double | `400.0` | stop time (free-fall units) |
| `dt_start` | double | `0.02` (RBC) | `dt` cap; adaptive `dt = min(MaxCFL/speed, dt_start)` |
| `MaxCFL` | double | `0.5` | CFL target |
| `n_steps` | int | `200000` | step cap (whichever of `t_end`/`n_steps` hits first) |

## `[init]`
| Key | Type | Default | Meaning |
|---|---|---|---|
| `perturb` | double | `0.02` (RBC) / `0.05` (DHVC) | cold-start perturbation amplitude |

## `[output]`
| Key | Type | Default (RBC / DHVC) | Meaning |
|---|---|---|---|
| `nmonitor` | int | `200` / `500` | steps between `monitor.csv` lines |
| `t_stat_start` | double | `7500.0` / `30.0` | physical time to begin accumulating statistics |
| `t_stat_out` | double | `2500.0` / `30.0` | physical-time interval between `stats_t*.dat` files |
| `dir_stats` | string | `apps/<case>/statistics/` | output directory (put on shared NFS for GPU runs) |

## `[restart]`
| Key | Type | Default | Case | Meaning |
|---|---|---|---|---|
| `dir_in` | string | `""` | both | restart/warm-start source; empty = cold start |
| `dir_out` | string | `""` | both | checkpoint destination; empty = no checkpoint |
| `nrestart_write` | int | `0` | RBC | periodic checkpoint interval in steps (`0` = final only); alternates `ckpt_a`/`ckpt_b` |
| `stats_reset` | int | `0` | RBC | `1` = restart the running Θc/Nu averages fresh from this point |
| `warmstart_io` | string | `slab` | DHVC | `slab` (Tier-1 distributed read) \| `replicate` (root-reads-all + Bcast) |
| `warmstart_dryrun` | int | `0` | DHVC | `1` = dump interpolated-field checksums, then exit |

## `[bc.<axis>.<side>]` — boundary conditions
Parsed by `boundary::load_problem_from_config`. `<axis>` ∈ `x/y/z`, `<side>` ∈ `minus/plus`.
Only present when the axis is `"wall"`.
| Key | Type | Meaning |
|---|---|---|
| `T` | double | Dirichlet wall temperature (RBC: `+0.5`/`-0.5`) |
| `ghost` | string | scalar ghost policy, e.g. `"antisymmetric"` |
| `ghost_uv` | string | cell-centered U/V wall ghost policy, e.g. `"zero"` |
