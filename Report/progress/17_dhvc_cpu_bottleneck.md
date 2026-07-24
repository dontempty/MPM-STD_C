# DHVC CPU 성능 병목 분석 보고서 (2026-06-15)

> 대상: MPM-STD_C **CPU 경로** (라이브러리 `src/.../cpu/`, 앱 `apps/dhvc/main.cpp`). 병렬은 **MPI-only**(OpenMP/스레드 없음 → 랭크당 1코어).
> 기준 케이스: DHVC Ra=1e6, 384×96×192(=7.08M 셀), **np 4×4×3 = 48랭크**, cpu06.
> 방법: 코드 전수 분석(병렬 프로브) + 적대적 검증 + 로그 step-rate. ⚠ **CPU 경로엔 per-phase 타이머가 없어** 단계별 ms/step은 *측정값이 아닌 추론값*(collective 개수 + 메모리 대역폭 하한으로 도출).
> 짝 보고서: GPU 병목은 [16_dhvc_1e7_gpu_bottleneck.md](16_dhvc_1e7_gpu_bottleneck.md).
> ⚠ 분석 산출물입니다. **소스 코드는 수정하지 않았습니다.**

---

## 0. 결론 (Executive Summary)

CPU 병목은 **성격이 다른 두 가지**이며, 절대 혼동하면 안 된다.

| | **(A) 스텝당 비용** | **(B) 스텝 수 / 수렴** |
|---|---|---|
| 정체 | **latency-bound MPI collective** (분산 ADI TDMA + pressure 전치 + z-TDMA) | **명시적(explicit) 대류 → CFL dt 붕괴** |
| 증상 | 랭크 늘리면 오히려 악화(anti-scaling) → np 4×4×2=32가 sweet spot, 64+에서 pressure all-to-all 단독 >1s/step | Ra=1e7에서 dt가 1.5e-5로 붕괴 → ~200만 스텝 → **수 주** |
| 영향 | 1e6는 CPU로 완주 가능(~405 ms/step, 70k step) | **1e7은 CPU로 사실상 불가** (이것이 1e7을 GPU로 옮긴 진짜 이유) |
| 해법 | sweet spot 분할 + collective 중첩/병합 | **완전 음해(implicit) 대류** (논문 MPM-STD 스킴) |

**핵심:** GPU의 헤드라인 병목(필드 전체 host-staging)은 **CPU엔 존재하지 않는다** — CPU halo는 경계면만 교환(face-only, 저렴). CPU의 per-step 병목은 sweet spot에서는 비교적 가볍다. 하지만 **두 백엔드 모두 (B)의 CFL 스텝-수 벽을 공유**하며, GPU는 per-step 비용만 줄일 뿐 스텝 수는 그대로다. 따라서 **음해 대류만이 1e7을 근본적으로 푼다(양쪽 백엔드 공통).**

---

## 1. 기준 측정 (유일한 실측 앵커)

`log/fig7_1e6_29254.out`: `host=cpu06 ntasks=48`, 08:31:02 → 16:23:13 = **28,331초 / 70,000 step = 404.7 ms/step**. 격자 384×96×192 = 7.08M 셀. 실행 스크립트 `apps/dhvc/fig7_run48.sbatch` → `mpirun -np 48 ... input_fig7_1e6n48.toml`(np1=4, np2=4, **np3=3** = 48). **깨끗한 48랭크·1랭크/코어 실행**(oversubscription 아님). 수렴: Re_dstar=10.968 vs Ng 11.008, div~9e-14 → 1e6는 **완주**.

> ⚠ 단계별 ms/step 타이머는 CPU에 없음(`main.cpp`엔 모니터용 `MPI_Allreduce`뿐). 아래 per-step 분해는 **collective 개수·통신자 크기·메모리 대역폭 하한으로 추론**한 것이며, 절대 ms는 미측정이다.

---

## 2. (A) 스텝당 비용 병목 — latency-bound MPI collective

코드에서 *확실히* 셀 수 있는 것은 "스텝당 collective의 종류·개수·통신자 크기"다. 페이로드가 작아(KiB~수 MiB) **대역폭이 아니라 지연(latency)이 지배**한다.

### 스텝당 통신 인벤토리 (코드 직접 카운트)

| 단계 | 스텝당 collective | 통신자 | codeRef |
|---|---|---|---|
| **분산 ADI TDMA** (momentum 9 + scalar 3 sweep) | 각 sweep이 자기 축 comm에서 분산 TDMA. **주기축 X/Y → `solve_many_cyclic`**, **벽축 Z → `solve_many`**. np1=4,np2=4,np3=3이라 세 축 모두 분할됨 | per-axis (4·4·3) | `momentum_cpu.cpp:92,96`; `scalar_cpu.cpp:61,65`; `pascal_tdma_many.cpp:185-209` |
| **Pressure 전치(transpose) 4회** | C→I, I→C = 2×`MPI_Alltoall`(X-comm np1) ; I→Y, Y→I = 2×`MPI_Alltoallv`(Y-comm np2). ~4.5 MiB/step | np1=4, np2=4 | `pressure_engine_cpu.cpp:514,552,612,667` |
| **Pressure 분산 z-TDMA** | `solve_tdma_z_`가 `solve_many`를 **2회(real+imag)**, 각 4 Ialltoallw → **8 Ialltoallw/step**. **comm_xz = np1·np3 = 12**(파이프라인 최대 sub-comm) | comm_xz=12 | `pressure_engine_cpu.cpp:773-776`, `:254` |
| **Face-only halo** | `sync_field_cpu` ~10회/step. 각 = **6 `MPI_Sendrecv`**(경계면 slab만, subarray 데이터타입). **필드 전체 복사 없음** | nearest-neighbor | `subdomain.cpp:119-147`; `momentum_cpu.cpp:156-164` |
| **전역 Allreduce 2회** | CFL MAX(`cfl_cpu.cpp:30`) + pressure-mean SUM(`pressure_engine_cpu.cpp:326`). div 모니터는 500스텝마다 | full comm(48) | 작음(8B, O(log P)) |

### 적대적 검증으로 바로잡은 점 (정확도 중요)
- **Pressure 내부에서 전치(4)보다 z-TDMA(8 Ialltoallw)가 더 큰 통신 부하다.** "전치가 지배"는 너무 좁은 프레이밍 — 정확히는 **z-TDMA ≳ 전치**, 둘 다 latency-bound. `comm_xz`가 가장 큰 sub-comm(=np1·np3)이라 z-TDMA가 핵심.
- 종합본 초안의 "ADI가 48 Ialltoallw/step"은 **과장**. X/Y는 주기축이라 `solve_many_cyclic`(non-cyclic `solve_many` 아님). 분산 TDMA는 자기 축이 분할됐을 때만 통신하며 정확한 collective 수는 cyclic 구현에 따름 — "세 축 전부 48회"로 단정 불가.
- **serial compute는 ~5–15%(추정)**, 미측정. 7.08M/48 ≈ 147k 셀/랭크, 7-pt 스텐실 working set ~1.2MB → 코어당 메모리 대역폭(10–20 GB/s) 바운드. **유일하게 랭크 늘리면 줄어드는 항**.

### 왜 sweet spot(anti-scaling)이 생기나
compute는 ~1/ranks로 줄지만, 각 collective 비용 ≈ `α·(P_axis−1) + β·bytes`에서 **bytes가 작아 `α·(P−1)` 지연항이 지배**하고, 동기 collective의 *개수*도 늘어난다. 즉 랭크를 늘리면 **싼 것(compute)은 줄고 비싼 것(latency collective)은 곱으로 는다.** → 벽시계가 내려가다 바닥 치고 **다시 상승**.
- **np 4×4×2 = 32**(comm_xz=8) = sweet spot. **48**(comm_xz=12) 느려짐. **≥64**에서 pressure all-to-all 단독 >1s/step. `np3=1/4`는 "진행 안 함" — `comm_xz = np1·np3`가 8 z-TDMA Ialltoallw/step을 직접 짊어지기 때문(np3을 키우면 악화).

**(A) 정리:** sweet spot에서 per-step은 **통신 지연 지배**(추정 ~85–95%)이며, pressure 경로의 **z-TDMA(comm_xz)와 전치**가 핵심, 분산 ADI TDMA가 동급. compute는 소수.

---

## 3. (B) 스텝 수 병목 — Ra=1e7의 진짜 킬러 (검증: CONFIRMED)

per-step 최적화와 **무관하고 훨씬 큰** 벽.

### 근본 원인 — 대류가 RHS에서 명시적(explicit)
- 모멘텀: `compute_mpmstd_rhs`가 3방향 `div(u·q)` 전체를 `conv_coeff·div(u·q)`로 가산, **`conv_coeff = −conv_factor·dt`, `conv_factor=1.0`**(전체 가중). ADI 밴드는 **0.25·dt 반음해 자기대류**만 가짐(부분). `momentum_kernels_cpu.cpp:43,79-83,114`; `momentum_cpu.cpp:27`
- 스칼라: `add_convection_rhs`가 `rhs += −dt·div(u·φ)` 완전 명시적. 음해 밴드는 `0.5·dt·α·L_d` 확산만. `scalar_kernels_cpu.cpp:94`; `scalar_cpu.cpp:35`

### 결과 — dt가 순수 대류 CFL(점성항 없음)에 묶임
`cfl_cpu.cpp:24-33`: `speed = max(|u|/dx1 + |v|/dx2 + |w|/dx3)`, `dt = min(MaxCFL/speed, dt_cap)`. tanh-stretch z-격자(γ3=2.6, 벽 근처 미세 dz)에서 속도가 오르면 벽법선 항이 폭발.

### 정량화 (검증: `apps/dhvc/logs/cn_1e7.log`, Ra=1e7, ν=2.65e-4, 256×128×192, np 4×4×2)
```
step=500 dt=4.09e-4 → 1000 dt=4.83e-5 → 2000 dt=3.2e-5 → 3000 dt=1.52e-5  (t는 겨우 0.352)
```
- 발달(t≈30–40)까지 **~2.0–2.6M step**(dt가 더 떨어지므로 200만은 *하한*).
- 1e7 격자(1e6의 ~9.4배)에서 2M step × ~0.4s(1e6 비용 가정) ≈ **9.3일**, 실제 격자 더 크므로 **수 주**. (step 3000에서 취소됨.)
- 대조: **1e6은 ν가 3× 커서 dt가 ~1.7e-3로 회복** → 70k step에 t=99 완주(`fig7_1e6_29254.out`). 이 ~100배 지속 dt 차이가 1e6 가능 / 1e7 불가를 가른다.

### 해법 = 논문 스킴
참조 MPM-STD는 **완전 음해 대류**를 써서 대류 CFL을 제거하고 dt를 *정확도*로 정한다. **1e7을 GPU로 옮긴 이유가 바로 이것.** 단, **GPU는 per-step 비용만 줄이고 스텝 수(~2M)는 그대로** → 음해 대류는 **두 백엔드 모두**에 필요한 유일한 근본 해법. (참고: [14_momentum_fidelity_vs_fortran.md](14_momentum_fidelity_vs_fortran.md) §3.2가 누락된 `−M·Uⁿ` 잔차항·명시 대류를 이미 지적.)

---

## 4. CPU vs GPU 병목 대조

| | **CPU (48랭크)** | **GPU (2×A100, np3=2)** |
|---|---|---|
| per-step 병목 | latency-bound **MPI all-to-all** (ADI TDMA + pressure 전치 + z-TDMA); 32랭크 넘으면 anti-scale | **host-staging PCIe**: 필드 전체 D2H/H2D @ ~2.09 GB/s ≈ 스텝의 55% ([Report 16](16_dhvc_1e7_gpu_bottleneck.md)) |
| halo | **face-only**, 저렴, scaling 양호 (`subdomain.cpp:119-147`) | **필드 전체** D2H/H2D — **GPU 전용 군더더기**, 비쌈 |
| step-count(1e7) | ~2M step → 수 주 | **동일 ~2M step** (GPU는 횟수 못 줄임) |

→ GPU의 헤드라인 문제(필드 전체 staging)는 **CPU엔 없음**. CPU의 헤드라인 문제(all-to-all 지연)는 sweet spot에선 상대적으로 가벼움. **공통 벽은 CFL 스텝-수.**

---

## 5. 권고 (효과 큰 순 — 제안만, 코드 미수정)

| 순위 | 권고 | 위치 | 효과 |
|---|---|---|---|
| **1** | **완전 음해(implicit) 대류 구현** — `div(u·q)`/`div(u·φ)`를 RHS에서 ADI 음해 연산자로 이동, CFL을 대류 무관하게 (델타형 `−M·Uⁿ` 잔차항 동반) | `momentum_kernels_cpu.cpp:43,79-83,114`; `scalar_kernels_cpu.cpp:94`; `cfl_cpu.cpp:24-33` | **~2M step 벽 제거 → 1e7을 CPU·GPU 양쪽에서 실현.** (B) 전체 해결. 압도적 1순위 |
| **2** | **sweet spot 분할 + collective 중첩/병합** — np 4×4×2=32(최소 comm_xz)로, np3 키우지 말 것; 전치·Ialltoallw를 compute와 overlap(이미 nonblocking, `Waitall` 즉시 호출 대신 독립 작업 채우기); CFL-MAX와 pressure-mean Allreduce 2회를 1회로 융합 | `pressure_engine_cpu.cpp:254,514-667,773-776,326`; `pascal_tdma_many.cpp:188-206`; `cfl_cpu.cpp:30` | (A) per-step 통신 지연 완화; anti-scaling 회피 |
| 3 | (선택) **랭크 내 스레딩** — 스텐실/Thomas 루프 스레드화 → 코어 많고 랭크 적게 → all-to-all 참여자 감소 | `momentum_kernels_cpu.cpp`, `scalar_kernels_cpu.cpp`; `Makefile.inc`(현재 `-fopenmp` 없음) | compute ~5–15%만 + MPI-only 설계 복잡화 → 후순위 |

---

## 6. 한 줄 요약

> CPU는 **필드 전체를 복사하지 않는다**(halo는 경계면만, GPU 군더더기 없음). CPU 병목은 (A) **all-to-all MPI 지연**(랭크 늘리면 악화 → np 4×4×2=32 sweet spot)과 (B) **명시적 대류가 부르는 CFL dt 붕괴**(1e7에서 dt=1.5e-5 → 수 주)다. (A)는 sweet spot에서 가볍고, **진짜 벽은 (B)** — 이는 GPU도 공유하므로 **완전 음해 대류**가 1e7을 CPU·GPU 모두에서 푸는 유일한 근본 해법이다.

---

## 7. 참조 파일 (절대경로)

- `src/equation/pressure/cpu/pressure_engine_cpu.cpp:254,514,552,612,667,773-776,326` — 전치·z-TDMA·comm_xz·mean Allreduce
- `src/equation/momentum/cpu/momentum_kernels_cpu.cpp:43,79-83,114` · `momentum_cpu.cpp:27,92,96,156-164` — 명시 대류·분산 ADI·face halo
- `src/equation/scalar/cpu/scalar_kernels_cpu.cpp:94` · `scalar_cpu.cpp:35,61,65` — 명시 대류·분산 ADI
- `src/driver/cpu/cfl_cpu.cpp:24-33` — 순수 대류 CFL(점성항 없음)
- `src/parallel/mpi/subdomain.cpp:119-147` — face-only halo(6 Sendrecv, 필드 전체 복사 없음)
- `../PaScaL_TDMA_C/src/pascal_tdma_many.cpp:185-209` — 분산 TDMA Ialltoallw
- `log/fig7_1e6_29254.out` · `apps/dhvc/fig7_run48.sbatch` · `apps/dhvc/input_fig7_1e6n48.toml` — 48랭크 1e6 기준
- `apps/dhvc/logs/cn_1e7.log` — 1e7 dt 붕괴 증거
