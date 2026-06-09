# 구조 재설계 분석 — 시뮬레이션 상태 집약(aggregate)

> 요청: time loop 안에서 인자가 객체ref·포인터·스칼라로 뒤섞여 복잡함. 전처리(MPI/CUDA/subdomain/topology)는 유지하고, loop 밖에서 **하나의 객체**(어떤 변수·BC·init·파라미터를 쓸지)를 정의한 뒤 loop에서는 `equation::solve_momentum_cpu(problem)`처럼 **그 객체만 넘겨** 호출하고 싶다. 가능한지, 코드를 어떻게 만들어야 하는지, 문제는 어디인지 분석.

---

## 1. 지금 무엇이 복잡한가 (구체 진단)

현재 channel main loop의 대표 호출:
```cpp
equation::solve_momentum_cpu(mom, U, V, W, dU, dV, dW, g, problem, *tdma, sub, (real_t)nu, dt);  // 12 인자
equation::solve_pressure_cpu(poi, dt, U, V, W, P, g, problem, *tdma, sub);                        // 10 인자
```
인자 종류가 4가지로 뒤섞임 — 이게 복잡함의 정체:
| 종류 | 예 | 성격 |
|---|---|---|
| 가변 상태(객체 ref) | `mom`, `U,V,W,P`, `dU,dV,dW`, `poi`, `stats` | 매 스텝 바뀜 |
| 읽기전용 컨텍스트(객체 ref) | `g`(Grid), `problem`(BC), `sub`(Subdomain) | run 내내 고정 |
| **포인터 역참조** | `*tdma` (`make_default`가 `unique_ptr` 반환) | 보기 싫은 deref |
| 스칼라 | `dt`(매 스텝), `nu`(고정), `dpdx`(가변), `total_vol`(고정) | const/비const 혼재 |

→ "고정 컨텍스트 + 가변 상태 + 포인터 + 스칼라"가 한 호출에 평평하게 나열돼 **무엇이 입력이고 무엇이 갱신되는지, 무엇이 고정인지** 한눈에 안 들어옴.

## 2. 가능한가? — 가능하다

상태를 한 객체(`Case`/`Simulation`)에 모으고 loop에서 그 객체만 넘기는 건 CFD에서 흔한 `Solver`/`Case` 패턴이다. 다만 "**얼마나** 모으느냐"에 따라 설계가 갈리고, 여기에 **핵심 긴장**이 있다.

## 3. 핵심 긴장 — 여기가 "문제"의 본질

이 리팩토링이 시작된 이유는 교수님 피드백 **"추상화 과다 → 가독성 저하"** 였고, rev.2는 그래서:
- **연산 = 자유함수(데이터를 명시 인자로)** → 호출부만 봐도 데이터 흐름이 보이게
- **쓰는 것만 선언/할당** (등온 채널은 `T`/`mu`를 아예 선언 안 함)

그런데 "**모든 걸 한 객체에 담고 `f(obj)`로만 호출**"은 정반대 방향으로 갈 위험이 있다:
- `solve_momentum_cpu(sim)` 한 줄만 보면 **momentum이 `sim`의 무엇을 읽고 무엇을 쓰는지 안 보임** (god-object → 데이터 흐름 은폐). ← 교수님이 싫어한 바로 그 지점
- 한 타입에 모든 필드(T, mu…)를 넣으면 **"쓰는 것만 할당"이 깨짐**
- 제네릭 방정식(channel/cavity/dhvc/rbc 공용)이 **앱별 객체**에 묶이면 재사용성 하락

즉 목표(호출 간결화)는 옳지만, **순진한 god-object는 리팩토링의 원래 목적을 되돌린다.** 좋은 설계는 "간결한 호출"과 "데이터 흐름 가시성·재사용성"을 **둘 다** 잡아야 한다.

## 4. 풀어야 할 구체 문제 7가지

1. **이름 충돌**: `Problem`은 이미 = `boundary::Problem`(BC 기술자, `core::Boundary`). 새 집약 객체는 다른 이름이어야 함 (`Case`/`Simulation`/`Run`/`Channel`).
2. **설정 vs 가변 상태 분리**: "어떤 변수·BC·init"(설정, run 내내 고정)과 필드/시스템(매 스텝 변함)은 성격이 다름. 한 객체에 둘 다 넣을지, 나눌지.
3. **쓰는 것만 할당(rev.2)**: 고정 단일 `Case`에 T·mu를 항상 넣으면 등온 케이스에서 낭비. → **앱별 Case**(channel엔 T·mu 없음, rbc엔 있음)로 풀어야 함.
4. **제네릭 방정식 vs 앱별 객체**: `solve_momentum_cpu`는 4개 앱 공용이어야 하는데, 앱별 `ChannelCase`를 직접 받으면 공용성이 깨짐. → 방정식은 **공용 조각**(필드/시스템/컨텍스트)을 받아야 함.
5. **수명·소유**: 집약 객체가 grid/bc/tdma/필드를 소유. `PressureSystem.engine`(shared_ptr)은 grid/bc/tdma **참조를 보유** → 멤버 순서·이동(move) 시 dangling 주의.
6. **CPU/GPU 타입 분리(C1)**: 가변 상태는 `CpuField`/`GpuField` 두 벌 → 집약 객체도 `CpuCase`/`GpuCase`(또는 템플릿). 컨텍스트(grid/bc/tdma/sub)는 host 단일이라 공유.
7. **포인터 정리**: `unique_ptr<TdmaRegistry>`의 `*tdma` deref → 집약 객체가 보유하고 **ref/accessor로 노출**하면 호출부에서 사라짐.

## 5. 세 가지 설계안

| 안 | 호출 모습 | 장점 | 단점 |
|---|---|---|---|
| **A. 순수 god-object** (`f(sim)`, sim이 모든 걸 가짐) | `solve_momentum_cpu(sim)` | 호출 최단 | 데이터흐름 은폐(교수님 반대점), "쓰는 것만 할당" 붕괴(또는 앱별 타입↔제네릭 충돌), 단위테스트에 sim 전체 필요, const 안전성↓ |
| **B. 컨텍스트 번들 + 명시 상태** (read-only `Domain` + 가변 인자) | `solve_momentum_cpu(dom, mom, U,V,W, dU,dV,dW, dt)` | 12→7인자, 포인터/혼재 제거, **데이터흐름 가시**, 제네릭 유지 | 여전히 여러 인자(레시피 1줄은 아님) |
| **C. 현행 유지** | 12인자 | 완전 투명 | 너무 장황(=불만 지점) |

→ A는 원하는 호출이지만 리팩토링 취지를 거스름. C는 불만. **B + "앱 레벨 thin wrapper"의 조합**이 원하는 main 모습을 주면서 취지도 지킴(아래 권장).

## 6. 권장 설계 — 앱 집약(Case) + 얇은 step 래퍼 + 제네릭 자유함수(읽기전용 Domain)

3계층으로 나눈다:

**(1) 읽기전용 컨텍스트 `core::Domain`** — god-object가 아님(가변 물리상태 없음, "where/how"만):
```cpp
namespace mpmstd::core {
struct Domain { const Grid& grid; const Boundary& bc; TdmaRegistry& tdma; const Subdomain& sub; };
}
```

**(2) 제네릭 방정식 자유함수** — `Domain` + 자기가 만지는 **가변 상태만** 명시(데이터흐름 보임, 4앱 공용):
```cpp
void solve_momentum_cpu(const core::Domain& d, core::MomentumSystem& mom,
                        const CpuField& U, const CpuField& V, const CpuField& W,
                        CpuField& dU, CpuField& dV, CpuField& dW, real_t nu, real_t dt);
```

**(3) 앱 집약 `Channel` + 얇은 step 래퍼** — "한 객체에 모든 정보", main은 `step(case)` 스타일:
```cpp
// apps/channel/channel.hpp — 이 앱이 쓰는 것만! (T/mu 없음 → 쓰는 것만 할당 유지)
struct Channel {
  core::Grid grid; core::Boundary bc; TdmaRegistryPtr tdma; const core::Subdomain& sub;  // 소유/보유
  core::CpuField U,V,W,P, dU,dV,dW;
  core::MomentumSystem mom; core::PressureSystem poi; post::Stats stats;
  real_t nu, dpdx, dt, total_vol; /* + forcing/IO 설정 */
  core::Domain domain() { return {grid, bc, *tdma, sub}; }   // 뷰로 즉석 생성(자기참조 저장 회피)
};
// 얇은 래퍼(앱별, 한 번만 작성):
void momentum_step(Channel& c) {
  auto d = c.domain();
  assemble_momentum_const_visc_cpu(d, c.mom, c.U,c.V,c.W, c.nu, c.dt);
  solve_momentum_cpu(d, c.mom, c.U,c.V,c.W, c.dU,c.dV,c.dW, c.nu, c.dt);
  update_velocity_cpu(c.U,c.V,c.W, c.dU,c.dV,c.dW);
  sync_field_cpu(c.V, c.bc.V, c.sub); sync_field_cpu(c.W, c.bc.W, c.sub);
}
```
그러면 **main의 time loop이 원하는 모습**이 된다:
```cpp
Channel c = build_channel(cfg, sub, ...);   // loop 밖: "어떤 변수·BC·init·파라미터" 여기서 결정
while (c.time < c.t_end && ...) {
  c.dt = cfl_dt(c);
  momentum_step(c);
  forcing_step(c);
  pressure_step(c);
  post_step(c);
}
```
- "어떤 변수/BC/init를 쓸지"는 **`build_channel(...)`(생성자/팩토리)** 가 결정 = 한 곳에 모음 ✓
- main loop은 `step(c)`만 호출 ✓ (원하는 간결함)
- 그 아래 제네릭 자유함수는 **무엇을 읽고/쓰는지 시그니처에 그대로** → 교수님 취지·재사용성 유지 ✓
- 포인터 deref·인자 혼재는 래퍼 안으로 숨고 main에서 사라짐 ✓
- 등온이면 `Channel`에 T/mu 없음, rbc는 `Rbc`에 T/mu/buoy 포함 → 쓰는 것만 할당 ✓
- GPU는 `CpuChannel`/`GpuChannel`(또는 템플릿), `Domain`은 host 단일 공유 ✓

핵심 차이(권장 vs 순수 A): **집약 객체는 "소유·구성·수명"을 담당**하고, **데이터 흐름은 제네릭 자유함수 시그니처에 남긴다.** 래퍼가 둘을 잇는다.

## 7. 만약 정말 `equation::solve_momentum_cpu(case)` (단일 인자)를 원한다면
가능하지만 §3·§4의 대가(데이터흐름 은폐, 제네릭↔앱별 충돌, 쓰는 것만 할당 붕괴)를 받아들여야 한다. 절충: 제네릭 함수는 §6대로 두고, 앱에서 `momentum_step(case)`(=원하는 단일 인자 스타일)만 노출. 실질적으로 동일한 사용감.

## 8. 마이그레이션 단계 (작게, 검증하며)
1. `core::Domain` 도입(뷰 struct) + `tdma`를 deref-pointer 대신 보유/accessor화.
2. 방정식/물리/post 자유함수 시그니처를 `(Domain, …가변상태…)`로 정리(P1에서 정한 시그니처 1차 변경 — 한 묶음씩, 매번 Re_tau=180 회귀 비트-동일 확인).
3. `apps/channel/Channel` 집약 + `build_channel` + step 래퍼 작성 → main loop을 `step(c)`로 교체.
4. cavity/dhvc/rbc는 같은 패턴의 자기 집약 타입 사용(쓰는 것만).

**리스크**: 시그니처 재변경이라 회귀를 깨기 쉬움 → 단계마다 스모크+회귀로 비트-동일 확인. 자기참조(Domain이 Channel 멤버 ref) dangling 주의 → Domain을 **저장하지 말고 뷰로 즉석 생성**.

---

## 결론
- **가능하다.** 단, "모든 걸 한 객체에 담아 `f(obj)`"의 **순진한 형태(A)** 는 리팩토링이 없애려던 과추상화·데이터흐름 은폐를 되살린다 — 그게 유일한 "문제".
- **권장(B+래퍼)**: main loop은 원하는 `step(case)` 간결함을 얻고, 그 아래 제네릭 자유함수는 `(Domain + 명시적 가변상태)`로 데이터 흐름·재사용성·"쓰는 것만 할당"을 지킨다. 포인터/혼재 인자는 사라진다.
- 이름은 `Problem`(BC 전용) 말고 `Channel`/`Case`/`Simulation`을 쓸 것.
