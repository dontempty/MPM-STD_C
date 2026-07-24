# C++ CFD Solver Architecture Design (MPI + GPU + LES-Oriented)

## 1. Overview

본 문서는 GPU 및 MPI 기반의 고성능 C++ 유동 해석 솔버를 개발하기 위한 객체지향 구조 설계안을 정리한 문서이다.

설계 목표는 다음과 같다.

```text
1. 다양한 유동 문제(Channel, RBC, LES 등)를 확장 가능하게 구성
2. GPU 성능을 최우선으로 고려
3. MPI 기반 domain decomposition 지원
4. 불필요한 변수 및 방정식 생성 최소화
5. 유지보수성과 확장성 확보
6. 수치 계산 kernel은 절차적으로 유지
```

핵심 설계 철학은 다음과 같다.

```text
Field   = 데이터 저장
Model   = 계수 및 source 계산
Equation= 방정식 풀이
Solver  = 전체 흐름 제어
Kernel  = 실제 계산 수행
MPI     = 통신 수행
GPU     = 장치 및 메모리 관리
```

---

# 2. 전체 구조

전체 구조는 다음과 같이 구성한다.

```text
CFDCode
├─ main.cpp
├─ Config
├─ Runtime
│  ├─ MPIManager
│  ├─ GPUManager
│  └─ Logger
│
├─ Mesh
│  ├─ GlobalMesh
│  ├─ LocalMesh
│  └─ HaloInfo
│
├─ Field
│  ├─ FieldRegistry
│  ├─ ScalarField
│  └─ VectorField
│
├─ BoundaryCondition
│
├─ Model
│  ├─ LESModel
│  ├─ BuoyancyModel
│  └─ PropertyModel
│
├─ Equation
│  ├─ MomentumEquation
│  ├─ PoissonEquation
│  └─ EnergyEquation
│
├─ LinearSolver
│  ├─ TDMA
│  ├─ PoissonSolver
│  └─ KrylovSolver
│
└─ Solver
   ├─ BaseSolver
   ├─ ChannelSolver
   └─ RBCSolver
```

---

# 3. main.cpp 구조

`main.cpp`는 최대한 얇게 유지한다.

```cpp
int main(int argc, char** argv)
{
    MPIManager mpi(argc, argv);

    Config config("input.yaml");

    GPUManager gpu(config, mpi);

    std::unique_ptr<BaseSolver> solver;

    if (config.caseType == "channel") {
        solver = std::make_unique<ChannelSolver>(config, mpi, gpu);
    }
    else if (config.caseType == "rbc") {
        solver = std::make_unique<RBCSolver>(config, mpi, gpu);
    }

    solver->initialize();
    solver->solve();
    solver->finalize();

    return 0;
}
```

main의 역할은 다음과 같다.

```text
1. MPI 초기화
2. 입력 파일 읽기
3. GPU 설정
4. Solver 생성
5. Solver 실행
```

---

# 4. Solver 계층

Solver는 전체 해석 흐름을 관리한다.

```text
Solver의 역할:
- Mesh 생성
- Domain decomposition
- Field 생성 및 allocate
- Equation 생성
- Model 생성
- Time loop 수행
- Output 관리
```

기본 구조는 다음과 같다.

```cpp
class BaseSolver {
public:
    virtual void initialize() = 0;
    virtual void solve() = 0;
    virtual void finalize() = 0;

protected:
    Config& config;
    MPIManager& mpi;
    GPUManager& gpu;

    Mesh mesh;
    FieldRegistry fields;
};
```

---

# 5. initialize() 구조

초기화는 반드시 단계별로 분리한다.

```cpp
void Solver::initialize()
{
    initializeMesh();
    initializeMPI();
    initializeFields();
    allocateFields();
    initializeBoundaryConditions();
    initializeModels();
    initializeEquations();
    initializeLinearSolvers();
    initializeIO();
}
```

추천 순서는 다음과 같다.

```text
1. Mesh 생성
2. MPI decomposition
3. Field 종류 결정
4. Field 메모리 allocate
5. Boundary condition 생성
6. Model 생성
7. Equation 생성
8. Linear solver 생성
9. Output 설정
```

---

# 6. Field 구조

## 6.1 역할

Field는 유동 변수 데이터를 저장한다.

예:

```text
Vector field:
- U = (u,v,w)

Scalar field:
- p
- T
- rho
- mu
- nu_t
```

Field는 실제 데이터를 저장하며, Equation은 이를 참조만 한다.

---

## 6.2 FieldRegistry

필요한 변수만 동적으로 생성하기 위해 `FieldRegistry`를 둔다.

```cpp
class FieldRegistry {
public:
    void addScalar(const std::string& name);
    void addVector(const std::string& name);

    ScalarField& scalar(const std::string& name);
    VectorField& vector(const std::string& name);

private:
    std::unordered_map<std::string, ScalarField> scalarFields;
    std::unordered_map<std::string, VectorField> vectorFields;
};
```

---

## 6.3 Channel 예시

```cpp
fields.addVector("U");
fields.addScalar("p");
fields.addScalar("nu_t");
```

필요한 방정식:

```text
- MomentumEquation
- PoissonEquation
```

사용하지 않는 항목:

```text
- T
- EnergyEquation
- BuoyancyModel
```

---

## 6.4 RBC 예시

```cpp
fields.addVector("U");
fields.addScalar("p");
fields.addScalar("T");
fields.addScalar("nu_t");
```

필요한 방정식:

```text
- MomentumEquation
- PoissonEquation
- EnergyEquation
```

---

# 7. GPU 메모리 구조

GPU 성능을 위해 field 내부는 host/device 메모리를 동시에 가진다.

```cpp
class ScalarField {
public:
    double* hostPtr();
    double* devicePtr();

private:
    double* h_ptr;
    double* d_ptr;
};
```

실제 계산은 GPU pointer 기반으로 수행한다.

```cpp
computeMomentumKernel<<<grid,block>>>(
    U.devicePtr(),
    p.devicePtr(),
    rhs.devicePtr()
);
```

---

# 8. Equation 구조

Equation은 방정식을 푸는 객체이다.

```text
Equation
├─ MomentumEquation
├─ PoissonEquation
└─ EnergyEquation
```

각 Equation은 필요한 field만 참조한다.

---

## 8.1 MomentumEquation

MomentumEquation은 속도장을 계산한다.

필요 변수:

```text
- U
- p
- nu 또는 mu
- nu_t
- buoyancy source
```

구조 예시:

```cpp
class MomentumEquation {
public:
    virtual void solve() = 0;
};
```

---

## 8.2 문제별 분리

Channel과 RBC는 source term과 coupling이 크게 다르므로 분리한다.

```text
MomentumEquationBase
├─ ChannelMomentumEquation
└─ RBCMomentumEquation
```

공통 kernel은 내부에서 재사용한다.

---

# 9. LES 구조

LES는 `nu_t`를 계산하여 field에 저장한다.

```text
LESModel
   ↓
update nu_t
   ↓
MomentumEquation이 읽어서 사용
```

추천 구조:

```text
Model = coefficient producer
Equation = coefficient consumer
```

예:

```cpp
class SmagorinskyModel {
public:
    void update();
};
```

MomentumEquation:

```cpp
mu_eff = mu + rho * nu_t;
```

---

# 10. Buoyancy 구조

부력 모델은 별도 객체로 분리한다.

```text
BuoyancyModel
├─ Boussinesq
└─ VariableDensity
```

역할:

```text
T 또는 rho를 읽음
→ body force 계산
→ MomentumEquation에서 사용
```

RBC에서는:

```text
EnergyEquation
   ↓
T update
   ↓
BuoyancyModel
   ↓
MomentumEquation
```

---

# 11. Boundary Condition 구조

각 변수는 자기 boundary condition을 가진다.

```text
U:
- no-slip
- periodic
- inlet/outlet

p:
- Neumann
- periodic

T:
- fixed hot/cold
- adiabatic
```

구조:

```cpp
bc.add("U", NoSlip, YMinus);
bc.add("U", NoSlip, YPlus);

bc.add("p", Neumann, YMinus);
bc.add("p", Neumann, YPlus);

bc.add("T", FixedValue, YMinus, T_hot);
bc.add("T", FixedValue, YPlus, T_cold);
```

GPU에서는 BC마다 전용 kernel을 사용한다.

```text
apply_no_slip_kernel
apply_periodic_kernel
apply_fixed_T_kernel
```

---

# 12. MPI 구조

MPI는 Field가 직접 담당하지 않는다.

```text
Field = 데이터 저장
HaloExchange = 통신 수행
Solver = 통신 시점 결정
```

구조:

```cpp
class MPIManager {
public:
    int rank() const;
    int size() const;
};
```

```cpp
halo.exchange(fields.vector("U"));
halo.exchange(fields.scalar("T"));
```

---

# 13. GPU 구조

GPU 성능을 위해:

```text
상위 구조:
- 객체지향

하위 계산:
- 절차적 kernel
```

즉:

```text
CPU side:
- Solver
- Equation
- Model
- FieldRegistry

GPU side:
- flat array
- raw pointer
- procedural CUDA kernel
```

---

# 14. Kernel 설계 원칙

GPU kernel 내부에서는 다음을 피한다.

```text
- virtual function
- unordered_map lookup
- string 기반 접근
- 과도한 if branch
```

대신:

```text
- flat array
- contiguous memory
- compile-time specialization
- branch 최소화
```

를 사용한다.

예:

```cpp
__global__
void computeMomentumRHS(
    double* u,
    double* v,
    double* w,
    const double* p,
    const double* nu_t,
    double* rhs
)
{
    // stencil 계산
}
```

---

# 15. Time Loop 구조

## 15.1 Channel flow

```text
while time < endTime:

1. Halo exchange(U)
2. LES update
3. Momentum solve
4. Poisson solve
5. Velocity correction
6. Apply BC
7. Output
8. Advance time
```

---

## 15.2 RBC

```text
while time < endTime:

1. Halo exchange(U,T)
2. Energy solve
3. Apply T BC
4. Buoyancy update
5. LES update
6. Momentum solve
7. Poisson solve
8. Velocity correction
9. Apply