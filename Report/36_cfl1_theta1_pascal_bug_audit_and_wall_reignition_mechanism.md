# 36. PaScaL_TCS 대비 버그 감사 + CFL=1.0/θ=1.0 검증 — 윗벽 재점화 메커니즘 실측 확정

> **작성**: 2026-07-15
> **동기**: PaScaL_TCS는 프로덕션에서 `MaxCFL=1.0`으로도 안정적으로 도는데, MPM-STD_C의 RBC NOB
> 경로는 `MaxCFL=0.3`으로 묶여있어 런타임이 오래 걸린다. PaScaL_TCS/src/Report의 버그 목록을
> MPM-STD_C에 대조해 실제 이식 버그를 찾아 고치고, θ=1(완화 없음) 고정 상태에서 CFL=1.0이
> 안전한지 실측으로 검증한다.

---

## 1. PaScaL_TCS 버그 9건 대조 결과

PaScaL_TCS/src/Report(21개 파일, 모두 "버그를 찾았다"는 코드리뷰 문서일 뿐 **PaScaL_TCS 자체에는
하나도 수정 안 됨**, `git show HEAD` 기준)를 MPM-STD_C의 GPU 우선(NOB/var 경로)·CPU 순으로
대조했다.

| # | 버그 | MPM-STD_C 상태 | 조치 |
|---|---|---|---|
| 1 | `solvedU` own-X 점성 밴드: `Mu(im,j,k)` 대신 `Mu(i,j,k)`를 양쪽 face에 사용 | **동일 버그 존재** (own-Y/Z는 이미 두 셀을 올바르게 구분하는데 own-X만 `muc`를 양쪽에 씀; CPU 주석에 "ref quirk"로 의도적 보존까지 돼있었음) | **수정** |
| 2 | `solvedV`의 `jvp=jS_BC(jm)`/`jvm=jS_BC(jp)` 벽마스크 스왑 | 해당 없음 — 포트는 명시 마스크 대신 반대칭 고스트 미러링 + `k_modify_tdma_row_gpu` 사후 밴드 폴딩을 씀. 벽속도=0인 현재 설정에서 두 메커니즘이 수학적으로 등가임을 직접 재검증(§3의 own-direction RHS/밴드 자기일관성 확인 포함) | 조치 불필요 |
| 3 | ADI 스테이지 간 `mu1,mu2`/`mu3,mu4`/`km` 등 stale 지역변수 재사용 | GPU 커널은 매 launch마다 스레드별로 전역메모리에서 새로 계산(스테이지 간 상태 없음), CPU도 스테이지별 루프 안에서 매번 새로 선언 — **구조적으로 발생 불가** | 조치 불필요 |
| 4 | `solvedW`의 `zbc_down`/`zbc_up`에서 `dmx2(jp)`/`dmx2(j)` 스왑 | 포트에는 이 해석적 벽유속 항 자체가 없음(고스트 미러링으로 대체) — 벽속도=0인 한 dormant, 움직이는 벽에는 원래도 없던 RHS 보정 gap(별건) | 조치 불필요(기존에 알려진 gap과 동일) |
| 5 | `mpi_momentum_coeffi`의 `Mu(:,j,k)=...` 콜론 슬라이스가 마지막 고스트값을 전체 라인에 브로드캐스트 | C++ 쪽은 셀별로 개별 대입이라 이런 슬라이스 구문 자체가 없음 — **구조적으로 발생 불가** | 조치 불필요 |
| 6 | `module_solve_thermal.f90:379` A4 대각계수가 이중으로 절반 처리됨(paper Eq.20 대비) | **동일 버그 존재** (`scalar_var_stencil.hpp`/`scalar_var_kernels_cpu.cpp` 양쪽) | **수정** |
| 7 | 스테이지 2/3에서 `invRhoCpCt_half`(값)을 `dinvRhoCpCt_half`(미분)이 필요한 자리에 사용 | **동일 버그 존재**, CPU 주석에 "note: invRhoCpCt_half, NOT dinvRhoCp... a faithful reference asymmetry"로 의도적 보존까지 돼있었음 | **수정** |
| 8 | 열솔버도 스테이지 간 stale `jm/jp/km/kp` | 버그 3과 동일 이유로 구조적으로 발생 불가 | 조치 불필요 |
| 9 | 압력 `dPhat` 벽고스트가 projection 후 안 갱신되어 `ExtraTerm`이 스퓨리어스 값 사용 | 이미 이번 세션 앞부분(Report/34급 수정)에서 `sync_field_gpu`/`apply_ghost_gpu` 호출로 해결돼 있었음 (`pressure_gpu.cu`, `pressure_engine_cpu.cpp`) | 이미 수정됨 |

### 1.1 실제 수정 파일 (버그 1, 6, 7)

- `src/equation/momentum/gpu/momentum_gpu.cu`: `k_mom_rhs_var` own=X 블록(`mACI`/`mAPI`/`mAMI`가
  이제 `mu[c]`(plus면)와 `mu[cim]`(minus면)을 구분), `k_mom_adi_var`의 own 분기(`mmu`가 조건부
  fallback 없이 항상 `mu[gi-ostride]`).
- `src/equation/momentum/cpu/momentum_var_kernels_cpu.cpp`: 동일 위치 대응 수정
  (`compute_mpmstd_rhs_var` own=X 블록, `build_adi_bands_var`의 `emit` 람다).
- `src/equation/scalar/scalar_var_stencil.hpp` (GPU/CPU 공유 헤더): stage-1 `ac` 이중절반 제거,
  stage-2/3 `ac`의 `irc_h`→`dirc_h` 교체.
- `src/equation/scalar/cpu/scalar_var_kernels_cpu.cpp`: 동일 수정.

**검증**: `make cpu` + `make BACKEND=cuda gpu` 둘 다 빌드 성공. `input_verify_buoy_ab.toml`(32×32×48,
3000스텝) 스모크테스트에서 CPU/GPU 두 백엔드가 사실상 동일 궤적(div~1e-15~1e-16, umax/wmax 소수점
단위까지 일치)으로 완주 — 회귀 없음 확인.

---

## 2. 새 기능: 순간(instant) Theta_c/Nu 모니터링

`gpu_print_step_rbc`(`apps/rbc/main_gpu_nob.cu`)가 기존엔 `nstat_start` 이후의 **누적평균**을
찍어서, 과도기 추이를 보려면 매번 `(sum_b-sum_a)/(n_b-n_a)`로 손으로 역산해야 했다(이번 세션에서
E20~E24 내내 반복). `gpu_compute_instant_tc_wt` 헬퍼를 신설해 매 `nmonitor` 호출마다 **이번 스텝의
순간값**을 별도로 계산·출력하도록 변경 — `nstat_start` 이전에도(과거엔 `Theta_c=0.0000` 플레이스홀더만
찍힘) 실제 순간값이 처음부터 보인다. 장기평균 통계(`stats.txt`, 재시작 연속성)는 기존 `gpu_accumulate_rbc`
경로 그대로 유지되어 영향 없음.

---

## 3. E25 — ΔT=20K, 256³ 냉시작, CFL=1.0/θ=1.0 프로덕션 시도

`input_e25_deltat20_cfl1_theta1.toml`(perturb IC, `MaxCFL=1.0`, `dt_start=0.05`로 CFL 자유,
BLKRELAX 미설정=θ=1.0 고정), job 30240, gpu02, 70만 스텝 완주(약 15시간).

- **안정성**: 문제 없음. div 3~8×10⁻¹⁴대 유지, NaN/블로업 없음, dt는 0.006~0.008대에서 톱니파 없이
  안정.
- **물리**: θ_c = **−0.1018**(끝값), 참조/기존 20K 평형(E18/E19, +0.0647)과 **부호까지 반대**.
  로컬(구간별) 역산 결과 측정구간(step 400k→700k) 내내 감속 없이 계속 더 음수로 이동 —
  냉시작이라 물리시간이 t≈4760까지밖에 못 가서(E18/E19는 웜스타트 체인으로 t≈20168 누적) 대규모
  순환 패턴이 아직 정착 못한 것으로 판단, **버그가 아니라 미수렴**. §4에서 직접 검증.

---

## 4. 진단: E19 평형 상태에서 CFL=1.0/θ=1.0 이어돌리기

가설 검증을 위해 `restart_e19_stats_reset_h256`(이미 t≈20168까지 축적된 확정 평형, θ_c=+0.0647)에서
CFL=1.0/θ=1.0(수정된 코드)로 이어서 돌렸다(`input_diag_e19_cfl1_instant.toml`, job 30249).

- **1000스텝 시점**: θ_c=+0.0649~0.0650 — 평형 그대로 유지. → E25의 음수는 **버그도 CFL=1.0 자체의
  본질적 문제도 아니고 순수 냉시작 미수렴**이라는 게 확인됨.
- **~16000스텝 시점**: θ_c가 +0.0649→**+0.0597**로 서서히, 그러나 뚜렷하게 하락. Nu도 9.6대→
  **12.4~13.4**로 30~40% 상승. 하락은 처음 ~10000스텝에서 대부분 일어나고 이후 0.060 근방에서
  완만해지는 조짐 — 폭주는 아니지만 **CFL=0.3 대비 실제 정확도 손실(θ_c 약 −8%)**은 있음.

### 4.1 dt 자체는 거의 안 늘어남

같은(수정된) 바이너리, 같은 restart로 `MaxCFL=0.3`(`input_diag_e19_cfl03_check.toml`, job 30251)과
`MaxCFL=1.0`을 나란히 비교:

| | dt 범위 |
|---|---|
| MaxCFL=0.3 | 0.0058~0.0068 |
| MaxCFL=1.0 | 0.0059~0.0079 |

이론상 host_max(CFL 메트릭)가 같다면 CFL=1.0의 dt가 3.3배 커야 하는데 **거의 차이가 없다.**
(참고: umax는 CFL=1.0쪽이 2.5배 이상 크다(1.2~1.45 vs 0.50~0.52) — dt를 지배하는 게 벌크 유속이
아니라는 신호.)

### 4.2 근본 원인 실측 — host_max는 정확히 윗벽, 그리고 능동적으로 자라난다

`src/driver/gpu/cfl_gpu.cu`에 `MPMSTD_CFLDBG=1` 게이트 진단(`k_cfl_find_argmax`, atomicCAS 기반
"first-writer" 방식으로 host_max를 달성하는 셀의 (i,j,k) 기록)을 추가해 직접 확인:

```
[CFLDBG] host_max=51.3  at local(i,j,k)=(6,235,255) of (n1i,n2i,n3i)=(256,256,256)
```

`k=255 = n3i-1` — **정확히 윗벽(냉각벽) 바로 안쪽 셀**, Report/30이 찾은 체커보드 재점화 지점과
동일 위치. 게다가 이 값이 5스텝 동안 **51.3 → 72.9 → 100.7 → 129.3 → 151.1(약 3배)로 능동
성장**하고, 그에 맞춰 dt는 0.0195 → 0.0137 → 0.00993 → 0.00774 → 0.00662로 빠르게 붕괴한다.
즉 CFL을 풀어주는 즉시 윗벽 근처 무언가가 자라나 host_max를 밀어올리고, `dt=min(MaxCFL/host_max,
dt_cap)` 공식이 이를 자동으로 다시 눌러버리는 **음성 피드백**이다 — 이게 CFL=1.0/θ=1.0이 안 터지고도
속도 이득을 못 보는 이유다.

### 4.3 θ(BLKRELAX)로 성장 억제 확인 — 자연현상이 아니라 수치 아티팩트

같은 조건에 `MPMSTD_BLKRELAX=0.5`만 추가:

| step | θ=1.0 host_max | θ=0.5 host_max | θ=1.0 dt | θ=0.5 dt |
|---|---|---|---|---|
| 1 | 51.3 | 51.3 | 0.0195 | 0.0195 |
| 3 | 100.7 | 58.1 | 0.00993 | 0.0172 |
| 5 | 151.1 | 63.7 | 0.00662 | 0.0157 |

θ는 물리적 의미가 전혀 없는 순수 수치 완화계수(블록커플링 보정의 세기 조절)인데, 이 값 하나로
같은 자리의 성장 속도가 확연히 달라진다(3배 성장 vs 1.24배 성장). 자연스러운 물리 현상이라면 θ와
무관해야 하므로, **이 성장은 Report/30이 확정한 블록커플링 한계안정성(재점화) 메커니즘 그 자체이며
새로운 자연 현상이 아니다** — θ=1로 몇 개의 이식 버그(§1)를 고쳐도 이 구조적 약점 자체는 그대로
남아있다는 뜻.

---

## 5. 종합 결론

1. **PaScaL_TCS 버그 9건 중 3건(1,6,7)이 MPM-STD_C에도 실제로 이식돼 있었고 모두 수정 완료**
   (CPU+GPU 빌드/스모크테스트 통과). 나머지는 구조적으로 발생 불가하거나 이미 다른 메커니즘으로
   안전했거나 이미 수정돼 있었음.
2. **CFL=1.0/θ=1.0은 "안 터진다"는 의미에서는 안전**하지만, **속도 이득은 거의 없고**(dt가
   CFL=0.3 때와 비슷한 수준으로 자기제한됨), 장시간 유지 시 **θ_c가 약 8% 낮아지고 Nu가
   30~40% 높아지는 실제 정확도 손실**이 있다.
3. 원인은 **윗벽 근처 블록커플링 한계안정성(Report/30 재점화)이 여전히 살아있고**, CFL-적응형
   dt 계산이 우연히 이걸 폭주 직전에서 막아주고 있을 뿐이다(host_max 성장 → dt 자동 감소 →
   성장률 둔화, 완전한 정지는 아님).
4. θ<1(BLKRELAX)을 같이 쓰면 이 성장이 확연히 느려지고 dt도 훨씬 크게 유지된다(θ=0.5에서
   5스텝 시점 dt=0.0157, θ=1.0의 0.00662 대비 2배 이상). **진짜 스텝 가속은 "CFL=1.0 단독"이
   아니라 "CFL=1.0 + θ<1 조합"으로 가야 한다** — Report/31의 경험적 θ·dt≤0.005~0.0075 경계와
   정합적이며, 이제 그 경계가 왜 존재하는지 메커니즘까지 실측으로 확인됐다.

## 6. 남은 일

1. CFL=1.0 + θ(0.2~0.5대) 조합으로 장시간(수십만 스텝) 프로덕션 검증 — θ_c가 참조값(+0.0647)
   근처에서 안정적으로 유지되는지, dt가 실질적으로(2배 이상) 커지는지 확인.
2. §4.2의 CFLDBG 계측을 더 오래(수천 스텝) 돌려서 θ=1.0에서 host_max가 진짜 어딘가서
   평형(자기제한)에 도달하는지, 아니면 아주 느리게 계속 자라는지 확인.
3. E25(냉시작, 미수렴 상태로 방치)는 폐기하거나, 필요시 훨씬 더 긴 정착 구간을 거쳐 재검증.
4. `MPMSTD_CFLDBG` 진단 코드는 opt-in(env var 게이트)이라 기본 동작에 영향 없음 — 향후 재점화
   재조사 시 재사용 가능.

## 7. 관련 산출물

- 수정 파일: `momentum_gpu.cu`, `momentum_var_kernels_cpu.cpp`, `scalar_var_stencil.hpp`,
  `scalar_var_kernels_cpu.cpp`, `src/driver/gpu/cfl_gpu.cu`(CFLDBG, opt-in), `main_gpu_nob.cu`
  (instant 모니터링).
- 진단 런: `apps/rbc/input_e25_deltat20_cfl1_theta1.toml`(job 30240),
  `input_diag_e19_cfl1_instant.toml`(job 30249), `input_diag_e19_cfl03_check.toml`(job 30251),
  `input_diag_e19_cfl1_argmax.toml`(CFLDBG 계측).
