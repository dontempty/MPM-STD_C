# 0단계 검증 방법 — "로직 불변"의 수치 증명

- 작성일: 2026-07-24
- 대상: [`0_stage.md`](0_stage.md)의 삭제 37파일 + 헤더 5개 편집
- 관문 근거: [`01_deabstraction_user_friendliness_plan.md`](01_deabstraction_user_friendliness_plan.md) §7 "로직 불변 검증 전략"
- 결론: **동일 결정론 케이스의 최종 필드 덤프 8개가 삭제 전(HEAD) ↔ 삭제 후 byte-단위로 완전히 동일** → 로직 불변 증명.

---

## 1. 왜 "바이너리 md5 비교"는 관문이 아니었나 (실제로 밟은 함정)

처음엔 계획서 §7의 "CPU 빌드 byte 비교"를 **앱 바이너리 md5 비교**로 시도했다. 삭제 전 7개 앱의 md5를 기록해두고, 삭제 후 재빌드하여 비교:

```
cavity      : 동일  ✅
channel/dhvc/rbc/validate/rbc_cpu_ob/rbc_cpu_nob : md5 다름  ❓
```

**6/7이 달라졌지만 이것은 회귀가 아니다.** 원인 두 가지:

1. **레이아웃 변경 (정상):** 0단계에서 `PressureSystem`의 미사용 멤버 `transform`(구조체 크기·오프셋 변경)과 `TdmaSolver`의 미사용 가상 메서드 2개(vtable 축소)를 제거했다. 이는 순수 파일 삭제가 아니라 **미사용 멤버 제거**이므로, 그 타입을 쓰는 TU의 바이너리 레이아웃이 바뀐다 — **동작은 불변, 배치만 변경**. `cavity`가 유일하게 동일한 이유는 이 바뀐 구조(`PressureSystem`/TDMA 다형성)를 쓰지 않는 프레임워크 스텁이라서다.

2. **빌드 자체의 비재현성:** HEAD 원본 소스를 **다른 위치(worktree)에서 빌드한 validate 바이너리조차** 기존 baseline md5와 달랐다. 즉 **같은 소스라도 빌드 경로·시점이 다르면 md5가 달라진다**(경로 문자열 등 비결정 요소). 

> **교훈:** 바이너리 md5는 이 프로젝트에서 결정론 신호가 아니다. "삭제·rename만 한 단계에서 1비트라도 다르면 실수"라는 §7 문구는 **런타임 수치 출력**에 적용해야지 컴파일된 바이너리에 적용하면 거짓 양성(false alarm)을 낳는다.

---

## 2. 올바른 관문 — `validate` 하네스의 필드 덤프

`apps/validate`는 정확히 이 목적으로 만들어진 결정론적 회귀 하네스다:

- **serial** (`np1=np2=np3=1`) → MPI 리덕션 순서 비결정성 없음
- **고정 dt** (`dt_start=4.0e-3`), **200 스텝**, **결정론적 IC**, 균일 격자 32×32×48
- 최종 `U,V,W,T`를 참조 레이아웃으로 `dump_*_{U,V,W,T}.bin`(raw `double`)에 덤프
- 두 입력으로 **두 모멘텀 경로를 모두** 커버:
  - `input_current.toml` → const-visc 모멘텀 경로
  - `input_faithful.toml` → faithful/var-visc 모멘텀 경로
- 두 경로 모두 0단계에서 건드린 코드(레이아웃 바뀐 `PressureSystem`·TDMA vtable, 편집한 `system.hpp`·`tdma_solver.hpp` 등)를 통과한다.

logic이 정말 불변이라면, 이 결정론적 200스텝의 **최종 필드는 비트 단위로 같아야** 한다.

---

## 3. before/after 절차 (git worktree)

삭제 후 상태는 현재 작업 트리에 있고, 삭제 **전**(HEAD 원본)은 별도 git worktree로 재구성했다. 두 빌드 모두 시스템 OpenMPI(`/usr/bin/mpicxx`)로, Intel oneAPI **없이** 빌드.

```bash
R=/shared/home/wel1come1234/workspace/MPM-STD_C
cd $R

# ── NEW: 삭제 후 (현재 작업 트리) ────────────────────────────
make clean && make BACKEND=cpu all            # 7개 앱 링크, EXIT 0
mkdir -p val_new && cd val_new
$R/build/cpu/bin/apps/validate $R/apps/validate/input_current.toml
$R/build/cpu/bin/apps/validate $R/apps/validate/input_faithful.toml
# → dump_mpm_{current,faithful}_{U,V,W,T}.bin  (8개), div ~1e-16

# ── OLD: 삭제 전 (HEAD 원본을 worktree로) ────────────────────
git worktree add --detach ../old_head_wt HEAD  # bands.hpp·solve/ 등 원본 복원 확인
cd ../old_head_wt && make BACKEND=cpu apps      # EXIT 0
mkdir -p val_old && cd val_old
../build/cpu/bin/apps/validate $R/apps/validate/input_current.toml
../build/cpu/bin/apps/validate $R/apps/validate/input_faithful.toml

# ── 비교 ────────────────────────────────────────────────────
for f in dump_mpm_current_{U,V,W,T} dump_mpm_faithful_{U,V,W,T}; do
  cmp -s val_old/$f.bin val_new/$f.bin && echo "IDENTICAL $f" || echo "DIFFER $f"
done

git worktree remove --force ../old_head_wt      # 정리
```

---

## 4. 결과

```
IDENTICAL  dump_mpm_current_U.bin
IDENTICAL  dump_mpm_current_V.bin
IDENTICAL  dump_mpm_current_W.bin
IDENTICAL  dump_mpm_current_T.bin
IDENTICAL  dump_mpm_faithful_U.bin
IDENTICAL  dump_mpm_faithful_V.bin
IDENTICAL  dump_mpm_faithful_W.bin
IDENTICAL  dump_mpm_faithful_T.bin

>>> 8/8 필드 덤프 byte-동일 (old HEAD == new post-stage0)
```

런타임 발산도 양쪽 `div ~4.5e-16`(발산 0)으로 동일. **동일 결정론 시뮬레이션이 const·var 두 경로 모두에서 최종장을 비트 단위로 동일하게 생성 → 0단계는 계산 결과를 바꾸지 않았음이 증명됨.**

---

## 5. 왜 이 증명이 충분한가 (3중 방어)

1. **grep(-E) 참조 0:** 삭제·제거한 모든 것이 빌드 트리에서 미참조임을 확장 정규식으로 확인.
2. **빌드 성공 = 컴파일러의 미참조 확인:** 만약 사용되던 심볼/멤버를 지웠다면 링크가 실패했을 것(undefined reference / no member). 성공했다는 것은 "지운 것이 안 쓰였다"는 기계적 증거.
3. **필드 덤프 byte-동일:** 위 두 개는 "안 쓰였다"를, 이 항목은 "그래서 결과가 같다"를 직접 보인다.

미사용 구조체 멤버·미사용 가상 메서드 제거가 바이너리 레이아웃을 바꾸는 것은 정상이며, 레이아웃이 아니라 **수치 출력이 같음**이 로직 불변의 올바른 척도다.

---

## 6. 한계와 후속

- **GPU 미검증:** 로그인 노드는 `.cu` 컴파일 불가 → `make BACKEND=cuda`는 GPU 노드 sbatch 필요. GPU는 atomic 연산으로 run-to-run 비결정이므로, 검증 시 byte가 아닌 **허용오차 비교**(div_max·Θ_c·Nu·평균 프로파일 상대오차 1e-12~1e-10)를 써야 한다([`01_...plan.md`](01_deabstraction_user_friendliness_plan.md) §7-1).
- **자산화 권장:** 위 §3 절차를 `scripts/regression_three_cases.sh`로 스크립트화하면 1·2·3·4단계에서 그대로 재사용되는 회귀 관문이 된다. 3케이스(channel/DHVC/RBC)로 확장 시, 백색소음 IC를 쓰는 케이스는 시드를 고정하거나 `validate`처럼 결정론 IC 변형을 써야 before/after 비교가 성립한다.
