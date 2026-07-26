# Output formats

Every quantity is in **free-fall units** (velocity = `√(gβΔT H)`, length = `H`,
1 time unit = 1 free-fall time). Files land under `[output] dir_stats` /
`[restart] dir_out`. On GPU runs these must be on shared NFS.

## `monitor.csv` — per-`nmonitor`-step time series (root rank)

**RBC** (`post::RbcStatistics`) — header matches data (6 columns):
```
step,t,dt,div,Theta_c,Nu
```
| Column | Meaning |
|---|---|
| `step` | step index |
| `t` | physical time |
| `dt` | timestep |
| `div` | max \|∇·u\| (divergence residual, ~1e-12) |
| `Theta_c` | instantaneous centre-plane mean temperature |
| `Nu` | instantaneous Nusselt = `1 + ⟨wθ⟩/α` |

**DHVC** (`post::DhvcStatistics`) — ⚠️ the header row lists **9** columns but each
data row writes **11** (two trailing timing columns are not in the header — a
known cosmetic mismatch, preserved as-is):
```
# header:  step,t,dt,div,Re_dstar,Re_inst,KE,Nu,theta_rms
# data:    step,t,dt,div,Re_dstar,Re_inst,KE,Nu,theta_rms,wall_s,ms_per_step
```
| Column | Meaning |
|---|---|
| `Re_dstar` | Re based on the **time-averaged** displacement-thickness profile |
| `Re_inst` | Re from this step's instantaneous profile |
| `KE` | volume-averaged kinetic energy |
| `Nu` | wall-normal convective Nusselt = `1 + ⟨wθ⟩/α` |
| `theta_rms` | volume RMS of temperature |
| `wall_s` | cumulative wall-clock seconds (data only) |
| `ms_per_step` | recent ms/step since last monitor (data only) |

## Statistics profile files (root rank)

**RBC** — `stats_t<time>.dat` (every `t_stat_out`) and `stats_final.dat`. CSV with
two `#` comment lines then 9 columns per z-plane (cumulative time-average):
```
# RBC z-profile stats (step=…, n=…, Ra=…, DeltaT=…)
# Z,T_mean,T_rms,U_mean,U_rms,V_mean,V_rms,W_mean,W_rms
<zc>,<T_mean>,<T_rms>,<U_mean>,<U_rms>,<V_mean>,<V_rms>,<W_mean>,<W_rms>
```

**DHVC** — `stats_t<time>.dat` in **Tecplot POINT** format
(`post::write_statistics_cpu`):
```
VARIABLES = "Z" "Z_plus" "U_mean" "W_mean" "u_rms" "v_rms" "w_rms" "uw_stress" "P_mean"
ZONE T="Stats", I=<nz>, J=1, K=1, DATAPACKING=POINT
<z> <z+> <U_mean> <W_mean> <u_rms> <v_rms> <w_rms> <uw_stress> <P_mean>
```
All profiles are dt-weighted time-averages accumulated from `t_stat_start`.

## Restart files (`dir_out/`)

One **decomposition-independent** global C-order (k-fastest) `double` array per
field — restartable at any rank count or mesh (§10 of the readme):
```
U.bin V.bin W.bin T.bin P.bin      # each = n1m·n2m·n3m interior doubles; P = accumulated pressure P_acc
```
Written by root-gather + `fwrite` (`write_restart_gathered_cpu`, NFS-safe).

Metadata sidecar differs by case:
- **RBC** `stats.txt` (`post::Restart`, 17-sig-fig):
  ```
  step <int>
  time <double>
  tc_sum <double>   tc_n <long>   wt_sum <double>   # running Θc/Nu sums
  ```
- **DHVC** `meta.txt` (`post::DhvcRestart`) — carries the source **mesh** so a run
  on a different grid can cross-mesh warm-start:
  ```
  step <int>   time <double>
  n1m <int>   n2m <int>   n3m <int>   Ra <double>
  ```

RBC's periodic checkpoints alternate `dir_out/ckpt_a` and `dir_out/ckpt_b` so a
half-written checkpoint never clobbers the last good one.
