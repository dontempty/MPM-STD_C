# C++ 유동 해석 코드 객체지향 구조 설계안

## 1. 설계 목표

본 문서는 C++ 기반 유동 해석 코드의 기본 객체지향 구조를 정리한 문서이다.

목표는 다음과 같다.

```text
1. 유동 변수의 메모리 관리를 명확히 한다.
2. 방정식별 풀이 루틴을 분리한다.
3. 사용자가 원하는 해석 문제에 따라 Solver를 구성할 수 있게 한다.
4. Channel flow, RBC와 같이 서로 다른 물리 문제를 같은 구조 안에서 확장 가능하게 만든다.
5. GPU/HPC 확장을 고려하여 상위 구조는 객체지향적으로, 하위 계산 루틴은 절차적으로 유지한다.
```

본 설계에서 핵심이 되는 클래스는 다음 세 가지이다.

```text
1. Field class
2. Equation class
3. Solver class
```

---

## 2. 전체 구조 개요

전체 구조는 다음과 같이 구성된다.

```text
Solver
 ├─ FieldRegistry / FieldSet
 │   ├─ scalar fields
 │   │   ├─ p
 │   │   ├─ T
 │   │   ├─ rho
 │   │   └─ nu_t
 │   │
 │   └─ vector fields
 │       └─ U = (u, v, w)
 │
 ├─ Equation
 │   ├─ MomentumEquation
 │   ├─ PoissonEquation
 │   └─ EnergyEquation
 │
 └─ solve()
     ├─ momentum.solve()
     ├─ poisson.solve()
     └─ energy.solve()
```

즉, `Field` 계층은 변수를 저장하고, `Equation` 계층은 방정식을 풀며, `Solver` 계층은 어떤 변수를 만들고 어떤 방정식을 어떤 순서로 풀지 결정한다.

---

## 3. Field Class

### 3.1 역할

`Field` 클래스는 유동 해석에서 사용하는 변수들을 저장하는 클래스이다.

예를 들어 다음과 같은 변수를 저장할 수 있다.

```text
Vector variables:
- U = (u, v, w)

Scalar variables:
- p
- T
- rho
- mu
- nu_t
```

여기서 중요한 점은 모든 문제에서 모든 변수가 필요한 것은 아니라는 점이다.

예를 들어 Channel flow에서는 보통 다음 변수가 필요하다.

```text
Channel flow:
- u
- v
- w
- p
- nu_t
```

반면 RBC와 같은 자연대류 문제에서는 온도장이 필요하므로 다음 변수가 필요하다.

```text
RBC:
- u
- v
- w
- p
- T
- nu_t
```

따라서 `Field` 클래스는 모든 변수를 고정 멤버로 미리 선언하는 방식보다, 필요한 변수만 등록하고 메모리를 할당하는 구조가 적합하다.

---

### 3.2 FieldRegistry 개념

이를 위해 `FieldRegistry` 또는 `FieldSet` 클래스를 둔다.

```cpp
class FieldRegistry {
public:
    void addScalar(const std::string& name);
    void addVector(const std::string& name);

    ScalarField& scalar(const std::string& name);
    VectorField& vector(const std::string& name);

    bool hasScalar(const std::string& name) const;
    bool hasVector(const std::string& name) const;

private:
    std::unordered_map<std::string, ScalarField> scalarFields;
    std::unordered_map<std::string, VectorField> vectorFields;
};
```

예를 들어 Channel flow에서는 다음과 같이 field를 등록할 수 있다.

```cpp
fields.addVector("U");
fields.addScalar("p");
fields.addScalar("nu_t");
```

RBC에서는 다음과 같이 등록할 수 있다.

```cpp
fields.addVector("U");
fields.addScalar("p");
fields.addScalar("T");
fields.addScalar("nu_t");
```

즉, 에너지 방정식을 풀지 않는 경우에는 `T`를 만들지 않아도 된다.

---

### 3.3 Field 접근 방식

각 방정식 클래스는 `FieldRegistry`를 참조하여 필요한 변수에 접근한다.

예를 들어 Momentum equation은 다음과 같은 변수를 참조할 수 있다.

```cpp
class MomentumEquation {
public:
    MomentumEquation(FieldRegistry& fields)
        : U(fields.vector("U")),
          p(fields.scalar("p")) {}

private:
    VectorField& U;
    ScalarField& p;
};
```

LES를 사용하는 경우에는 `nu_t`도 참조한다.

```cpp
class MomentumEquation {
public:
    MomentumEquation(FieldRegistry& fields)
        : U(fields.vector("U")),
          p(fields.scalar("p")),
          nu_t(fields.scalar("nu_t")) {}

private:
    VectorField& U;
    ScalarField& p;
    ScalarField& nu_t;
};
```

이 구조의 장점은 다음과 같다.

```text
1. 변수의 실제 메모리는 FieldRegistry가 관리한다.
2. 각 Equation은 필요한 변수만 참조한다.
3. 같은 Field를 여러 Equation이 공유할 수 있다.
4. 불필요한 변수는 생성하지 않아도 된다.
```

---

## 4. Equation Class

### 4.1 역할

`Equation` 클래스는 각각의 지배방정식을 푸는 클래스이다.

기본적으로 다음과 같은 방정식 클래스를 둘 수 있다.

```text
Equation
 ├─ MomentumEquation
 ├─ PoissonEquation
 └─ EnergyEquation
```

각 클래스는 자기 방정식을 푸는 데 필요한 변수만 `FieldRegistry`에서 참조한다.

---

### 4.2 MomentumEquation

`MomentumEquation`은 속도장을 갱신하는 역할을 한다.

일반적으로 필요한 변수는 다음과 같다.

```text
필요 변수:
- U
- p
- nu 또는 mu
- nu_t
- body force
```

Channel flow에서는 평균 압력구배, 벽면 경계조건, 난류 점성계수 등이 포함될 수 있다.

```text
ChannelMomentumEquation:
- convection term
- diffusion term
- pressure gradient
- mean pressure gradient forcing
- wall boundary condition
- LES contribution
```

RBC에서는 부력항이 추가된다.

```text
RBCMomentumEquation:
- convection term
- diffusion term
- pressure gradient
- buoyancy force
- thermal coupling
```

따라서 Channel과 RBC의 momentum 방정식이 크게 다르면 하나의 `MomentumEquation`에 모든 분기를 넣기보다, 문제별로 분리하는 것이 좋다.

```text
MomentumEquationBase
 ├─ ChannelMomentumEquation
 └─ RBCMomentumEquation
```

공통 인터페이스는 유지한다.

```cpp
class MomentumEquationBase {
public:
    virtual void solve() = 0;
    virtual ~MomentumEquationBase() = default;
};
```

---

### 4.3 PoissonEquation

`PoissonEquation`은 비압축성 유동에서 압력 보정 또는 압력 Poisson 방정식을 푸는 역할을 한다.

일반적으로 필요한 변수는 다음과 같다.

```text
필요 변수:
- U
- p
- divergence of predicted velocity
```

Projection method를 사용하는 경우 전형적인 순서는 다음과 같다.

```text
1. Momentum equation으로 예측 속도 U* 계산
2. Poisson equation으로 압력 보정량 계산
3. 속도 보정
```

Poisson equation은 Channel flow와 RBC 모두에서 사용될 수 있다.

---

### 4.4 EnergyEquation

`EnergyEquation`은 온도장 또는 스칼라장을 푸는 방정식이다.

일반적으로 필요한 변수는 다음과 같다.

```text
필요 변수:
- T
- U
- thermal diffusivity
- heat source
```

RBC와 같은 자연대류 문제에서는 온도장이 부력항과 연결된다.

```text
EnergyEquation
   ↓
T update
   ↓
Buoyancy model
   ↓
MomentumEquation
```

반대로 에너지 방정식을 풀지 않는 Channel flow에서는 `EnergyEquation` 객체를 생성하지 않는다.

---

## 5. Solver Class

### 5.1 역할

`Solver` 클래스는 사용자가 원하는 해석 문제를 정의하고, 필요한 field와 equation을 구성하는 클래스이다.

즉, `Solver`는 다음을 결정한다.

```text
1. 어떤 변수를 선언할 것인가?
2. 어떤 방정식 객체를 생성할 것인가?
3. 각 방정식을 어떤 순서로 풀 것인가?
4. 시간 전진 루프를 어떻게 구성할 것인가?
```

---

### 5.2 Channel Solver 예시

Channel flow를 푸는 경우를 생각한다.

필요한 변수는 다음과 같다.

```text
Fields:
- U
- p
- nu_t
```

필요한 방정식은 다음과 같다.

```text
Equations:
- MomentumEquation
- PoissonEquation
```

에너지를 풀지 않으므로 다음은 생성하지 않는다.

```text
Not used:
- T
- EnergyEquation
- Buoyancy model
```

구조는 다음과 같다.

```cpp
class ChannelSolver {
public:
    ChannelSolver() {
        fields.addVector("U");
        fields.addScalar("p");
        fields.addScalar("nu_t");

        momentum = std::make_unique<ChannelMomentumEquation>(fields);
        poisson  = std::make_unique<PoissonEquation>(fields);
    }

    void solve() {
        while (time < endTime) {
            momentum->solve();
            poisson->solve();
            correctVelocity();
            advanceTime();
        }
    }

private:
    FieldRegistry fields;

    std::unique_ptr<MomentumEquationBase> momentum;
    std::unique_ptr<PoissonEquation> poisson;
};
```

---

### 5.3 RBC Solver 예시

RBC와 같은 자연대류 문제에서는 온도장이 필요하다.

필요한 변수는 다음과 같다.

```text
Fields:
- U
- p
- T
- nu_t
```

필요한 방정식은 다음과 같다.

```text
Equations:
- MomentumEquation
- PoissonEquation
- EnergyEquation
```

또한 부력 모델이 필요할 수 있다.

```text
Models:
- BuoyancyModel
```

구조는 다음과 같다.

```cpp
class RBCSolver {
public:
    RBCSolver() {
        fields.addVector("U");
        fields.addScalar("p");
        fields.addScalar("T");
        fields.addScalar("nu_t");

        buoyancy = std::make_unique<BoussinesqBuoyancyModel>(fields);

        momentum = std::make_unique<RBCMomentumEquation>(fields, *buoyancy);
        poisson  = std::make_unique<PoissonEquation>(fields);
        energy   = std::make_unique<EnergyEquation>(fields);
    }

    void solve() {
        while (time < endTime) {
            energy->solve();
            momentum->solve();
            poisson->solve();
            correctVelocity();
            advanceTime();
        }
    }

private:
    FieldRegistry fields;

    std::unique_ptr<BuoyancyModel> buoyancy;

    std::unique_ptr<MomentumEquationBase> momentum;
    std::unique_ptr<PoissonEquation> poisson;
    std::unique_ptr<EnergyEquation> energy;
};
```

---

## 6. Model Class와 Field의 관계

LES, 부력, 물성치 모델은 방정식 내부에 직접 공식을 박아 넣기보다 별도 모델로 분리하는 것이 좋다.

예를 들어 LES 모델은 `nu_t`를 계산한다.

```text
LESModel
   ↓
nu_t field update
   ↓
MomentumEquation
```

즉 LES 모델이 Momentum RHS를 직접 수정하는 것이 아니라, `nu_t` field를 업데이트하고 MomentumEquation이 이를 읽는 구조가 좋다.

```cpp
class LESModel {
public:
    virtual void update() = 0;
};
```

```cpp
class SmagorinskyModel : public LESModel {
public:
    void update() override {
        // U를 읽어서 nu_t 계산
        // fields.scalar("nu_t") 업데이트
    }
};
```

부력 모델도 마찬가지이다.

```text
BuoyancyModel
   ↓
T 또는 rho를 읽음
   ↓
buoyancy force 계산
   ↓
MomentumEquation이 RHS에 반영
```

즉 역할은 다음과 같이 나누는 것이 좋다.

```text
Field:
- 변수 저장

Model:
- 계수 또는 source term 계산

Equation:
- 방정식 조립 및 풀이
- 주요 field 업데이트

Solver:
- 전체 순서 제어
```

---

## 7. 분기 처리 원칙

같은 Momentum equation이라도 문제에 따라 계수와 source term이 달라진다.

예를 들어 다음과 같은 차이가 있다.

```text
Channel:
- mean pressure gradient forcing
- wall boundary condition
- LES eddy viscosity

RBC:
- buoyancy force
- thermal boundary condition
- temperature coupling
```

이때 모든 경우를 하나의 MomentumEquation 안에 `if`로 넣으면 코드가 복잡해진다.

비추천 구조는 다음과 같다.

```cpp
void MomentumEquation::solve() {
    if (isChannel) {
        // channel forcing
    }

    if (isRBC) {
        // buoyancy
    }

    if (useLES) {
        // LES
    }

    if (solveEnergy) {
        // thermal coupling
    }
}
```

추천 구조는 다음과 같다.

```text
1. 물리적으로 많이 다른 문제는 MomentumEquation 클래스를 분리한다.
2. 공통 계산 kernel은 공유한다.
3. 문제별 source term과 boundary condition만 분리한다.
```

즉 다음 구조가 적절하다.

```text
MomentumEquationBase
 ├─ ChannelMomentumEquation
 └─ RBCMomentumEquation

Common kernels:
- compute convection
- compute diffusion
- compute pressure gradient
- solve linear system
```

---

## 8. GPU/HPC 관점의 설계 원칙

GPU 성능을 고려하면 상위 구조와 하위 계산 루프를 분리해야 한다.

```text
CPU side:
- Solver class
- Equation class
- FieldRegistry
- Model selection
- Kernel launch control

GPU side:
- raw pointer
- flat array
- procedural kernel
- branch-minimized computation
```

즉, 객체지향은 solver 구성과 실행 흐름을 관리하는 데 사용하고, 실제 수치 계산 kernel은 단순한 배열 기반 함수로 구성하는 것이 좋다.

```cpp
computeMomentumRHS<<<grid, block>>>(
    U.devicePtr(),
    p.devicePtr(),
    nu_t.devicePtr(),
    rhs.devicePtr(),
    nx, ny, nz
);
```

GPU kernel 내부에서는 다음을 피하는 것이 좋다.

```text
- virtual function
- unordered_map 접근
- string 기반 field lookup
- 복잡한 class hierarchy
- cell loop 내부의 과도한 if 분기
```

대신 field 참조는 Equation 생성 시점에 미리 연결한다.

```cpp
class ChannelMomentumEquation {
public:
    ChannelMomentumEquation(FieldRegistry& fields)
        : U(fields.vector("U")),
          p(fields.scalar("p")),
          nu_t(fields.scalar("nu_t")) {}

    void solve() {
        launchChannelMomentumKernel(U, p, nu_t);
    }

private:
    VectorField& U;
    ScalarField& p;
    ScalarField& nu_t;
};
```

---

## 9. 권장 solve 흐름

### 9.1 Channel flow

```text
ChannelSolver::solve()

while time < endTime:

    1. LESModel.update()
       U -> nu_t

    2. MomentumEquation.solve()
       U, p, nu_t -> U*

    3. PoissonEquation.solve()
       U* -> p

    4. correctVelocity()
       U*, p -> U

    5. advanceTime()
```

---

### 9.2 RBC

```text
RBCSolver::solve()

while time < endTime:

    1. EnergyEquation.solve()
       U, T -> T

    2. BuoyancyModel.update()
       T -> buoyancy force

    3. LESModel.update()
       U -> nu_t

    4. MomentumEquation.solve()
       U, p, T, nu_t, buoyancy -> U*

    5. PoissonEquation.solve()
       U* -> p

    6. correctVelocity()
       U*, p -> U

    7. advanceTime()
```

순서는 수치 알고리즘에 따라 바뀔 수 있다. 중요한 점은 방정식끼리 서로 직접 호출하지 않고, `Solver`가 전체 순서를 관리한다는 것이다.

---

## 10. 최종 설계 요약

본 구조의 핵심은 다음과 같다.

```text
1. Field class는 변수 저장과 메모리 관리를 담당한다.
2. Equation class는 각 방정식의 조립과 풀이를 담당한다.
3. Solver class는 어떤 field와 equation을 사용할지 결정한다.
4. Channel flow에서는 U, p, nu_t 등을 생성하고 Momentum/Poisson equation만 사용한다.
5. RBC에서는 U, p, T, nu_t 등을 생성하고 Momentum/Poisson/Energy equation을 모두 사용한다.
6. LESModel은 nu_t field를 업데이트하고, MomentumEquation은 이를 읽어서 사용한다.
7. BuoyancyModel은 T 또는 rho를 이용해 body force를 계산하고, MomentumEquation이 이를 반영한다.
8. 문제별 momentum 방정식이 크게 다르면 ChannelMomentumEquation, RBCMomentumEquation처럼 분리한다.
9. 공통 수치 kernel은 공유하고, 문제별 source term과 boundary condition만 분리한다.
10. GPU 성능을 위해 실제 계산 루프는 절차적 kernel로 유지한다.
```

가장 중요한 설계 원칙은 다음과 같다.

```text
Field = 데이터
Model = 계수와 source term 계산
Equation = 방정식 풀이
Solver = 전체 해석 흐름 제어
```

이 구조를 사용하면 Channel flow, RBC, LES, 자연대류, scalar transport 등을 같은 코드 기반에서 확장할 수 있다.
