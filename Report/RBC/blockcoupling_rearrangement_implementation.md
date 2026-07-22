# 블록커플링 재배열 구현·검증 — MPM-STD_main 배치 적용 (재점화 수술)

> **작성**: 2026-07-15
> **선행 분석**: 같은 폴더 `mpmstd_main_vs_port_wall_reignition_analysis.md` (권고 ①의 구현)
> **한 줄 요약**: var/NOB 모멘텀 경로를 레퍼런스 MPM-STD_main 배치(solve 순서 U→V→W,
> 벽수직 dW 마지막 + 전량 RHS 선반영, 명시 사후보정은 수평 dV/dU만)로 재배열했다.
> **재점화의 능동(무한) 성장은 소멸**했고 물리(θ_c, Nu)는 CFL=1.0/θ=1.0에서 2000스텝 동안
> 평형값에 고정 유지. 다만 유계·θ-민감 근벽 모드가 남아 dt 이득은 θ=1.0 단독으론 미미 —
> **실전 가속은 여전히 CFL=1.0+θ=0.5 조합**(dt≈0.014, CFL=0.3 대비 약 2배)이며, 이제
> 그 조합이 레퍼런스 충실 배치 위에서 돈다.

---

## 1. 변경 내용 (수식은 전부 기존 검증 블록의 재조합 — 새 이산화 없음)

| 구성요소 | 이전 (PaScaL 전치 배치) | 이후 (MPM-STD_main 배치) |
|---|---|---|
| solve 순서 | dU → dW → dV | **dU → dV → dW (벽수직 마지막)** |
| `couple_dd_dV_var` | M21(dU) + M23(dW) 선반영 | **M21(dU)만** (ref solvedV 1211-1218) |
| `couple_dd_dW_var` | M31(dU)만 선반영 | **M31(dU) + M32(dV)** (ref solvedW 1524-1540) — dW로 가는 결합 전량이 z-ADI 암시필터 통과 |
| `block_couple_dW_var` (dW←dV 사후보정) | **존재 — 재점화 펌프** | **삭제** (수식은 M32로 이동) |
| `block_couple_dV_var` (dV←dW 사후보정) | 없음 (M23이 선반영에 있었음) | **신설** (ref blockLdV 1612-1688) — 수식은 구 dd_dV의 M23 블록 그대로 |
| `block_couple_dU_var` | dU←dV,dW 사후보정 | 변경 없음 (보정된 dV 사용, ref blockLdU 순서) |
| θ(BLKRELAX)/BLKMASK_TOP 훅 | dW·dU 사후보정 스테이지 | **dV·dU 사후보정 스테이지**로 이동 (진단 태그 dW.blk/g.blkW → dV.blk/g.blkV) |

수정 파일: `momentum_gpu.cu`(커널 8/9/10 + `solve_momentum_var_gpu`),
`momentum_var_kernels_cpu.cpp`, `momentum_cpu.cpp`(`solve_momentum_var_cpu`), `kernels.hpp`.
6개 교차블록(M21/M31/M32 선반영 + M23/M12/M13 사후보정) 커버리지는 이전과 동일 — 바뀐 것은
**어느 성분이 비여과 명시 보정을 받는가**뿐이다. 이제 벽수직 dW는 레퍼런스처럼 어떤 명시
사후보정도 받지 않는다. 참고: DHVC도 var 경로(μ≡1)를 쓰므로 이 변경의 영향권(회귀 확인 권장).

## 2. 검증

### 2.1 빌드·스모크
- `make cpu`(로그인 노드) + `make BACKEND=cuda apps`(GPU 노드, job 30261) 모두 성공.
- 스모크(`input_verify_buoy_ab.toml`, 32×32×48, 3000스텝, job 30263): CPU/GPU 궤적이 표시
  자릿수까지 일치(θ_c 동일, div GPU ~1e-15/CPU ~1e-16), NaN 없음, 완주.

### 2.2 판정 런 — E19 평형 restart + CFL=1.0/θ=1.0, 2000스텝 (CFLDBG, job 30264급)
`input_diag_blockfix_cfl1.toml`, `restart_e19_stats_reset_h256`(t≈20168, θ_c=+0.0647)에서:

| 항목 | 구코드 (Report/36 §4) | 신코드 (본 런) |
|---|---|---|
| host_max 초기 거동 | 51→73→101→129→151 **단조 능동 성장**, dt 붕괴 지속 | 51→118→156 (재시작 과도) 후 **감쇠 반전** 154→133→105, 이후 109~161 유계 진동 (평균 140) |
| dt (정착) | 0.0059~0.0079 | 0.0062~0.0092 (평균 0.0072) |
| θ_c | 1000스텝까진 유지, ~16k스텝에 **0.0597로 −8% 하락** | **0.0649~0.0650 고정** (2000스텝 전 구간, 하락 조짐 없음) |
| Nu (instant) | 9.6 → **12.4~13.4 (+30~40%)** | **9.55~9.71 안정** |
| div | — | 3~8×10⁻¹⁴ 정상 |

### 2.3 θ-민감성 A/B (신코드, 같은 restart, MPMSTD_BLKRELAX=0.5, 500스텝)

| | host_max (100스텝 이후) | dt |
|---|---|---|
| θ=1.0 | 109~161, 평균 140 | ~0.0072 |
| θ=0.5 | 52~82, 평균 67 | **0.0135~0.0167 (~0.014)** |

여전히 θ-민감 → 잔존 모드는 순수 물리가 아니라 (이제 수평 성분으로 옮겨간) 사후보정
채널의 유계 한계진동이다. 단, 구코드와 달리 (i) 무한 성장하지 않고, (ii) 물리량을
오염시키지 않는다(θ_c/Nu 무손상).

## 3. 판정과 운용 권고

1. **재점화(무한 능동 성장 → dt 자기붕괴 음성 피드백) 메커니즘은 제거됐다** — 벽수직
   증분이 비여과 명시 보정을 받는 경로 자체가 사라졌기 때문(레퍼런스와 동일 구조).
   CFL=1.0/θ=1.0에서 구코드가 보인 정확도 손실(θ_c −8%, Nu +30~40%)의 조기 신호가
   2000스텝 동안 전무하다.
2. **속도 이득 관점의 실전 조합은 CFL=1.0 + θ=0.5**: dt≈0.014로 CFL=0.3 프로덕션
   (0.0058~0.0068) 대비 **약 2배**. θ=1.0 단독은 잔존 유계 모드가 host_max를 ~140에
   붙들어 dt 이득이 거의 없다(구코드와 같은 결론이지만, 이제 θ<1이 "폭주 방지"가 아니라
   "잔존 한계진동 감쇠" 역할 — 안전 여유가 다르다).
3. **남은 확인 항목**: (i) θ=1.0 장시간(1만 스텝 이상)에서 θ_c 무손상 지속 여부 — 구코드
   드리프트는 ~10k스텝에 걸쳐 발현했으므로 2000스텝은 조기 신호 확인까지만 됨;
   (ii) CFL=1.0+θ=0.5 장시간 프로덕션 검증(Report/36 §6-1과 동일 프로토콜);
   (iii) DHVC(var 경로 공유) 회귀 — Fig7 ±2% 재확인.

## 4. 산출물

- 판정 런: `apps/rbc/input_diag_blockfix_cfl1.toml` + `scripts/rbc_diag_blockfix_cfl1.sh`
  (로그 `apps/rbc/logs/rbc_diag_blockfix_cfl1_full.log`)
- θ A/B: `apps/rbc/input_diag_blockfix_theta05.toml` + `scripts/rbc_diag_blockfix_theta05.sh`
- 컴파일: `scripts/compile_blockfix.sh`, 스모크: `scripts/smoke_blockfix.sh`
- restart 산출: `apps/rbc/restart_diag_blockfix_cfl1/`(step 2382000),
  `restart_diag_blockfix_theta05/`(step 2380500)
