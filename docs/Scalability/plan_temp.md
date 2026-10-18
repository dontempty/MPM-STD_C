# rbc_nob 최적화 — 현재 계획 (2026-10-18 정리)

`plan.md` 의 원 계획이 1~11회차로 끝난 뒤의 **남은 일**을 정리한 임시 계획서. 수치는 최신 바이너리(꼬리표 `v100pt`), V100 gpu02, 256³ np 1 기준.

## 1. 현재 위치

| 항목 | 값 |
|---|---|
| 스텝 시간 (256³ np 1) | 원본 198 → **87.5 ms (×2.26)** |
| 512³ np 2 / np 8 | 582 → 433 / 206 → 146 ms |
| weak 256³/rank np 8 | 210 → 107 ms (효율 54 → 82 %) |
| strong 256³ np 8 | 41 → 28.6 ms (효율 36 → 38 %; Alltoallv 9.4 ms = 33 %) |
| 다른 앱(dhvc·channel·BL·TBL) | 공용 경로 + 전용 RHS 빌드 적용, 256³ np 1 ×1.9~2.2 (`apps/<app>/plot.ipynb`) |
| 게이트 | 복사·융합류 = 비트 동일, 격자 역수류 = 허용오차(`check_gate.py`, 출력 정밀도 일치) |
| 커밋 | `simple` = `origin/simple` (`25eb77e`), 최적화 지점마다 `[최적화 날짜]` 주석, plan.md §10 표 |

남은 구성 (87.5 ms): momentum_solve 45.5 (그중 x·y 순환 TDMA 15.2 = 대역폭 바닥), scalar_solve 16.0, pressure_solve 7.5, momentum_rhs 7.0, pressure_rhs+correct 5.6, properties ×2 2.7, 나머지 3.2.

## 2. 지켜야 할 제약

- **3차원 분할(npx·npy > 1)이 나중에 들어온다** → x·y 가 랭크 안에 완결된다는 전제에 기대는 최적화(조립+순환 Thomas 융합)는 넣지 않는다. 레이아웃·밴드·커널 내부 최적화만.
- 측정은 1노드 8 GPU 상한, V100 만(H200 잡은 배정 불가). 4장 잡 두 개가 같은 노드에 동시에 들어가면 np 2·4 의 통신 항은 참고용.
- **잡이 대기·실행 중일 때 `build_gpu/bin/*` 를 다시 빌드하지 않는다** (런 시작 실패·바이너리 섞임). 꼬리표는 기존 것과 겹치지 않게.
- 코드를 복잡하게 만드는 작은 이득(2~3 %)의 변경은 넣지 않는다 (조각 분할 TDMA 전례). nsys/ncu 는 특정 커널을 손볼 때만.
- 결과(`out/`, `log/`, PNG)는 커밋하지 않는다. 노트북은 실행 결과 포함해 커밋.

## 3. 남은 후보 — 기대 단축 순 (256³ np 1, 스텝 87.5 ms)

| 순위 | 이벤트 | 현재 ms | 한계(대역폭) | 기대 단축 | 방법 | 게이트 / 위험 |
|---|---|---|---|---|---|---|
| 1 ✔ | pressure_rhs + pressure_correct | 5.6 → **4.1** | ~3.0 | 실측 **−1.5** (융합 −0.4 + 역수 −1.1) | 전용 커널 + 격자 역수 (`main_scalability.md` §2j) | 허용오차 — 완료 |
| 2 | pressure_solve 내부 | 7.5 | | **−2.7** | ① pack/unpack_bands(1.0) 제거 — 스펙트럼·밴드 배치가 같은 인덱스, 평균행 핀만 ② build_z(1.16) — PaScaL 변형 Thomas 가 a′·c′ 를 스크래치에 쓰면 모드 계수 한 번만 조립 ③ mean_remove(0.69) — DC 모드로 평균, scatter 에 빼기 융합 | ① 비트 동일·쉬움 ② PaScaL 로컬 커널(중간) ③ 게이지뿐 |
| 3 | scalar build_x·y | 7.1 | ~3.4 | **−2** | build_y(3.16) 가 타일 build_z(1.79) 의 1.8배 — 2-D 블록 매핑으로 통일 | 비트 동일 / 실험 |
| 4 | momentum 커플링 4개 | 5.8 | ~3.9 | −1.5 | ncu 로 L2 적중·레지스터 확인 뒤 launch_bounds·로드 순서 | 측정 후 판단 |
| 5 | properties_1 | 1.36 | 0 | −1.4 | clip 을 properties_2 앞으로, 다음 스텝 properties_1 삭제 | 클립된 셀만 차이 → 허용오차 |
| 6 | momentum_rhs | 7.0 | ~3.9 | −1.5 | 레지스터 118·점유율 25 % → launch_bounds·성분별 분리 | 불확실 |
| 7 | momentum build_x/y 의 D 복사 | 1.0 | | −1.0 | 순환 TDMA 커널에 d_in/d_out — 조립의 `D = source` 제거 | 비트 동일 / 작음 |
| 8 | dt_cfl + dt_viscous | 0.9 | ~0.35 | −0.4 | 리덕션 둘을 한 커널로 | 비트 동일 |

합계 ≈ −12.8 ms → **~75 ms (−15 %)**. 비트 동일·저위험(2①, 3, 7, 8)만 ≈ −5 ms. (1 은 완료: 87.5 → 86.4)

**바닥이라 안 건드리는 것**: momentum tdma_x+y 15.2(순환 Thomas 15패스 = 한계의 95 %), tdma_z 4.7, scalar tdma 6.6, build_z 5.4(타일, 78 %), cuFFT 1.4. np 8 의 Alltoallv 9.4 는 소켓 사이 링크 합계 한계(`main_scalability.md` §2g) — 3차원 분할이 풀 몫.

## 4. 진행 순서

1. **1 → 2① → 7 → 8** (전부 비트 동일, 하루 분량) → 빌드 → 스모크 게이트(동일) → V100 측정(새 꼬리표) → 노트북·문서.
2. **3** 실험: scalar build_x/y 를 2-D 블록 매핑으로 바꿔 np 1 만 먼저 재고, 이득이 10 % 미만이면 되돌린다.
3. **5**: clip 위치 이동 — 허용오차 게이트 + cv12 검증 덱 FINAL 비교.
4. **2②③**: PaScaL 스크래치 쓰기와 DC 모드 평균 — 각각 따로 게이트.
5. **4·6**: ncu 한 번 (couple_rhs_w, block_couple_u, build_rhs_kernel) → 레지스터·L2 수치를 보고 할지 결정.
6. 매 회차 끝: `build_notebooks.py` 재생성(꼬리표 추가), `main/momentum/scalar_scalability.md` 절 추가, plan.md §9·§10, 메모리.

## 5. 검증 계획 (물리 통계)

지금까지는 짧은 런(스모크 9행·200 스텝)의 동일성/허용오차 게이트뿐. 다음을 한 번 돌려 둔다:
- rbc_nob: cv12 검증 덱의 FINAL θ_c·Nu 를 원본 바이너리(worktree 빌드)와 현재로 비교.
- dhvc 1e5(짧게)·channel detcheck(Re_τ)·BL blasius_coarse(C_f vs Blasius)·TBL laminar_check(St) 의 FINAL 을 `build_gpu/bin/orig/<app>_scale` 과 비교.
- 기준: 반올림 수준(출력 정밀도 또는 1e-10 상대) 이내. 벗어나면 그 회차를 되돌린다.

## 6. 측정·시각화 도구 (있는 것)

- rbc_nob: `apps/rbc_nob/scalability/`(덱·잡·`build_notebooks.py`), 노트북 4개(`main/momentum/scalar_scalability.ipynb`, `present.ipynb` = V100 처음 vs 최신).
- 다른 앱: `tools/scale/`(README) — `instrument_main.py`(main.cpp → main_scale.cpp), `gen_decks.py`, `sbatch_scale.sh`, `build_app_plot.py` → `apps/<app>/plot.ipynb`. "전" 바이너리는 `git worktree add <dir> f330720` 로 재생성(저장소엔 최신 코드 한 벌만).
- main.cpp 를 고치면 main_scale.cpp 를 다시 생성한다.

## 7. 미결

- H200 결과는 10-17 중간 판(`h200h`)까지만 — 노트북의 H200 "후" 패널은 그 상태. 필요하면 H200 잡 재시도.
- 3차원 분할이 들어오면: 덱에 분할 모양(1×1×8, 2×2×2…) 축 추가, pressure 펜슬 전치와 함께 재배열 커널 재설계, np 8 통신 재측정.
