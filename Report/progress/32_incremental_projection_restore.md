# 32. RBC NOB 증분형(van Kan) 투영 복원 — 시간 1차 → 2차

> **작성**: 2026-07-07 (Report/31 §4c·§6의 후속)
> **동기**: dt 사다리(E14/E15/E17)에서 Nu가 dt에 ~1차로 수렴 (11.09/10.31/9.86 @
> dt 0.02/0.01/0.005; dt→0 외삽 ≈ 참조 9.61). BW delta-form은 명목 2차이므로 O(dt)
> 오차원이 있고, 유력 원인 = **포트 RBC의 비증분(Chorin) 투영**. PaScaL은 예측자에
> `−Cmp·invRhoc·∇Pⁿ`이 활성(mom.f90 331/343)인 **증분형(van Kan, 시간 2차)**.
> Report/28 §4.1의 "정상상태 실효 등가" 논증은 시간정확도 등가가 아니었음.

## 1. 변경 전 (비증분, 시간 1차)

- 예측자: ∇P 항 없음 → Poisson이 매 스텝 P 전체를 풂 (Var::P = 총압력 역할)
- 투영: u −= dt[∇P + (irc−1)∇P̂], P̂ = 2Pⁿ−…(총압력 외삽) — ExtraTerm 짝은 유지
- Chorin 비증분 분할의 속도 오차 = O(dt) — 관측된 1차 수렴과 일치

## 2. 변경 후 (증분형, PaScaL 구조)

스텝 구조 (main_gpu_nob.cu):
1. assemble + buoyancy 후: `add_pressure_gradient_nob_gpu` — **RHS −= dt·(1/ρ̂)_face·∇Pⁿ_acc**
   - invRhoc = PaScaL과 동일한 dx-가중 면 평균: 0.5·(dx(im)·invρ(i)+dx(i)·invρ(im))/dmx(i)
   - 신규 커널 `k_pred_pgrad_var` (pressure_gpu.cu). 기존 OB `k_pred_pgrad`는
     invRhoc 생략(OB에서 ≡1이라 무해; NOB에 쓰면 실버그 — Report/28에서 예고한 그 지점)
2. Poisson: 이제 자동으로 **증분 δp**를 풂 (예측자가 ∇Pⁿ을 이미 뺐으므로).
   dPhat 외삽·ExtraTerm 짝은 증분 기반이 되어 PaScaL 원형과 일치
3. 투영: u −= dt[∇δp + (irc−1)∇δp̂] (코드 변경 없음 — 의미가 증분으로 바뀜)
4. `accumulate_pressure_gpu`: **Pⁿ⁺¹ = Pⁿ + δp** (전 셀, 헤일로 포함 선형 누적)

## 3. restart 하위호환

- write_restart에 **P.bin**(누적 압력) 추가 (U/V/W/T/P + stats.txt)
- read_restart는 P.bin **있으면 읽고 없으면 0** — 구 restart와 호환. P=0 시작은
  self-starting (첫 Poisson이 P 전체를 반환 → 1스텝 만에 P_acc ≈ 총압력)

## 4. 영향 범위

- **rbc_gpu_nob 전용** (RBC NOB GPU 프로덕션 경로). DHVC GPU는 이미 증분형(무변경).
- **CPU RBC 경로(main_cpu_nob)는 비증분으로 남음** — CPU↔GPU 스킴 차이 발생
  (기존 스칼라 OB CPU/GPU 차이 메모와 같은 계열). CPU 정합 필요 시 동일 수술 이식.
- rbc_gpu_ob(OB)도 비증분 유지 — 검증 기준선 보존.

## 5. 검증 계획 (E18 사다리, 노드 가용 시)

restart_e14_h256(클린, t=17312) statzero에서 dt 0.02/0.01/0.005 × ~400 t.u.:
1. **차수 판정**: Nu(dt) 사다리 재측정 — 2차 회복이면 (11.09−10.31)/(10.31−9.86)
   비가 ~1.7 → ~4로 이동, dt 0.01에서 이미 Nu ≈ dt→0 극한(≈9.5)에 근접해야 함
2. **θ_c**: 평형값(현재 0.052 @ dt0.01 비증분)이 증분형에서 어디로 가는지 —
   압력-부력 균형의 시간오차 제거가 NOB 비대칭(χ 1.22 vs 1.30)에 주는 영향 확인
3. div, 안정성(θ·dt 곱 경계 재확인 — 예측자 압력항 추가로 블록커플링 이득이
   달라질 수 있음), 첫 스텝 P self-start 과도 확인
