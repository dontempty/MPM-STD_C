# tools/scale — 앱별 최적화 전/후 시간 측정

| 파일 | 역할 |
|---|---|
| `instrument_main.py` | `apps/<app>/main.cpp` → `apps/<app>/main_scale.cpp` (시간 루프의 호출 단위 문장을 PhaseTimer 구간으로 감싼 쌍둥이 main; `#ifdef MPMSTD_SCALE_MAIN` 가드라 일반 빌드에서는 빈 TU). main.cpp 를 고치면 다시 생성한다. `--no-solver-timers` = 풀이 내부 타이머·TDMA 통신 훅이 없는 옛 코드용. |
| `gen_decks.py` | `apps/<app>/scale/decks/input_{strong,weak}_256_np{1,2,4,8}.ini` (strong 256³, weak 256³/rank, 200 스텝, 통계·체크포인트 끔) |
| `sbatch_scale.sh` | 잡 스크립트: `APPS`, `VARIANTS="orig new"`, `NP_LIST`, `CASES`, `REPEAT`. CSV → `apps/<app>/scale/out/<case>_np<N>_<variant>_<job>_r<rep>.csv`, 환경 파일 `environment_<job>.txt` |
| `build_app_plot.py` | `apps/<app>/plot.ipynb` 생성·실행 (전/후 이벤트별 막대, scaling 곡선 겹치기, 감소량 + 요약표; 그림 저장 없음) |

빌드: `make GPU=1 GPU_ARCH=cc70,cc80,cc90 <app>_scale` → `build_gpu/bin/<app>_scale`.
"전" 바이너리 = 원본 커밋(f330720)의 worktree 에서 같은 방법으로 빌드해 `build_gpu/bin/orig/<app>_scale` 에 둔 것
(`git worktree add <dir> f330720`, `phase_timer.hpp`·Makefile·`instrument_main.py` 복사, `--no-solver-timers` 로 main_scale 생성).
환경변수: `MPMSTD_PHASE_START`(측정 시작 스텝, 기본 50), `MPMSTD_PHASE_CSV`(CSV 경로).
결과(`scale/out`, `scale/log`)와 그림은 커밋하지 않는다; `plot.ipynb` 는 실행 결과를 포함해 커밋한다.
