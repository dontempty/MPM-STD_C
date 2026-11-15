#!/usr/bin/env python3
"""rbc_nob_scalability 측정 덱 생성 — docs/Scalability/plan.md §3·§4.

  python3 gen_decks.py                            # 기본: n 256·512, strong·weak, slab·pencil, np 1..64
  python3 gen_decks.py --sizes 256 --np 1 2 4 8   # 일부만
  논문 세트 (decks_zyx/, 셀 크기 = weak_512 와 같음, pencil = cube 블록 zyx, slab = 같은 전역 격자를 1x1xnp 로;
  *_phases 는 --solver-phases --out decks_zyx_phases 를 더해 같은 명령; paper_campaign.md):
  python3 gen_decks.py --modes weak   --decomps pencil --pencil-order zyx --sizes 256 512 --np 1 2 4 8 16 32 64 128 256 512 --out decks_zyx
  python3 gen_decks.py --modes weak   --decomps slab   --pencil-order zyx --slab-mesh pencil --sizes 256 512 --np 8 16 32 64 128 256 --out decks_zyx
  python3 gen_decks.py --modes strong --decomps pencil --pencil-order zyx --strong-ref 512 --sizes 1024 --np 4 8 16 32 64 128 256 512 --out decks_zyx
  python3 gen_decks.py --modes strong --decomps pencil --pencil-order zyx --strong-ref 512 --sizes 2048 --np 16 32 64 128 256 512 --out decks_zyx
  python3 gen_decks.py --modes strong --decomps slab   --strong-ref 512 --sizes 1024 --min-local 4 --np 8 16 32 64 128 256 --out decks_zyx
  python3 gen_decks.py --modes strong --decomps slab   --strong-ref 512 --sizes 2048 --np 16 32 64 128 256 --out decks_zyx
    (GH200 120 GB: 512³/GPU ≈ 27 GB → 랭크당 2·512³ (54 GB) 까지 확실. 1024³ 는 np ≥ 4, 2048³ 는 np ≥ 32;
     2048³ np 16 (4·512³ ≈ 108 GB) 은 시험용 — OOM 이면 뺀다. strong_1024 slab np 256 은 nz_local 4 (하한 완화).)

  decks/<study>/<study>_np<N>.ini      study = <strong|weak>_<n>_<slab|pencil>
    strong : 전역 n³ 고정.
    weak   : 랭크당 n³ 고정 — 나눈 축 방향으로 전역 격자와 도메인을 같이 늘린다(셀 크기 유지).
             slab 은 z 만 늘어나므로 np 에 따라 변하는 것은 z 분산 TDMA 의 축소계뿐이다 (plan.md §3.4).
    slab   : npx = npy = 1, npz = np.
    pencil : np = 2^k 를 x → y → z 순서로 2 씩 나눈다 (2x1x1, 2x2x1, 2x2x2, 4x2x2, 4x4x2, 4x4x4).
    랭크당 셀이 축마다 MIN_LOCAL 보다 적어지는 조합(예: strong 256 slab np 64 → nz_local 4)은 만들지 않는다.
  decks/gate/gate_np<N>.ini + gate_ref_np<N>.ini — 같은 계산을 이 앱과 rbc_nob 으로 돌리는 짝
    (job_scal.sh 의 게이트: 두 런의 모니터 줄이 같아야 측정이 본런과 같은 계산이다).

물리는 cv12 ΔT 40 본런 덱(apps/rbc_nob/input_rbc_nob_cv12_dt40.ini)과 같다 — 측정하는 경로가 본런 경로다.
dt 는 비용과 무관(스텝당 일이 같다)하므로 sec/step 로만 비교한다.
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_LOCAL = 8   # 랭크당 축 셀 수 하한 (halo 2면이 25% 를 넘지 않게 — plan.md §3.3)
BASE_L = (6.0, 6.0, 1.0)   # strong 도메인 = cv12 덱의 6×6×1 박스

PHYSICS = """[stretch]
uniform_z = false
gamma_z = 2.6

[physics]
rayleigh = 1.0e6
prandtl = 2547.0
delta_temperature = 40.0

[initial]
perturb = 0.05
"""


def split(np_, decomp, order="xyz"):
    """np → (npx, npy, npz). pencil 은 order 의 축 순서로 2 씩 나눈다 (zyx → 1x1x2, 1x2x2, 2x2x2, 2x2x4, 2x4x4, 4x4x4)."""
    if decomp == "slab":
        return (1, 1, np_)
    if np_ & (np_ - 1):
        raise SystemExit(f"pencil 분할은 np = 2^k 만 받는다 (np {np_})")
    p = [1, 1, 1]
    axes = ["xyz".index(c) for c in order]
    k = 0
    while p[0] * p[1] * p[2] < np_:
        p[axes[k]] *= 2
        k = (k + 1) % 3
    return tuple(p)


def case(mode, n, decomp, np_, order="xyz", strong_ref=0, slab_mesh="slab", min_local=MIN_LOCAL):
    """(전역 격자, 도메인, 프로세스 격자) 또는 만들 수 없으면 (None, 이유)
       strong_ref > 0 이면 strong 도메인 = BASE_L × n/strong_ref — 셀 크기를 weak_<strong_ref> 와 같게 (논문 strong 세트).
       slab_mesh = "pencil" 이면 weak slab 의 전역 격자·도메인을 같은 np 의 pencil 과 같게 잡는다 (분할만 1x1xnp) —
       같은 문제를 slab / pencil 로 나눈 비교. 기본 "slab" 은 z 로만 늘리는 원래 규칙."""
    procs = split(np_, decomp, order)
    if mode == "strong":
        mesh = (n, n, n)
        length = tuple(l * n / strong_ref for l in BASE_L) if strong_ref else BASE_L
    else:
        grow = split(np_, "pencil", order) if (decomp == "slab" and slab_mesh == "pencil") else procs
        mesh = tuple(n * p for p in grow)
        length = tuple(l * p for l, p in zip(BASE_L, grow))
    for axis, (m, p) in enumerate(zip(mesh, procs)):
        if m % p:
            return None, f"{'xyz'[axis]} {m} 이 {p} 로 안 나뉜다"
        if m // p < min_local:
            return None, f"{'xyz'[axis]}_local {m // p} < {min_local}"
    groups = procs[0] * procs[1]   # 압력 펜슬 전치 조건 (pressure/solve.hpp): ny ≥ npx·npy, x 모드 수 ≥ npx·npy
    if mesh[1] < groups or mesh[0] // 2 + 1 < groups:
        return None, "압력 펜슬 전치 조건 (ny, nx/2+1 ≥ npx·npy)"
    return (mesh, length, procs), None


def deck_text(title, mesh, length, procs, time_block, output_block, extra=""):
    return f"""; {title}
; gen_decks.py 생성 — 손으로 고치지 말고 다시 생성할 것
[mesh]
nx = {mesh[0]}
ny = {mesh[1]}
nz = {mesh[2]}

[domain]
lx = {length[0]:.1f}
ly = {length[1]:.1f}
lz = {length[2]:.1f}

{PHYSICS}
[process]
npx = {procs[0]}
npy = {procs[1]}
npz = {procs[2]}

[time]
{time_block}
dt_max = 0.05
max_cfl = 0.3
viscous_coupling_number = 12

[output]
{output_block}
{extra}"""


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512], help="n (strong: 전역 n³, weak: 랭크당 n³)")
    ap.add_argument("--modes", nargs="+", default=["strong", "weak"], choices=["strong", "weak"])
    ap.add_argument("--decomps", nargs="+", default=["slab", "pencil"], choices=["slab", "pencil"])
    ap.add_argument("--np", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64], dest="nps")
    ap.add_argument("--steps", type=int, default=200, help="max_step (기본 200)")
    ap.add_argument("--warmup", type=int, default=50, help="timing.warmup_steps (기본 50 → 150 스텝 누적)")
    ap.add_argument("--monitor", type=int, default=50, help="output.monitor_interval")
    ap.add_argument("--solver-phases", action="store_true",
                    help="[timing] solver_phases = true (풀이 안의 호출별 구간까지; 경계 동기화가 늘어 느려진다)")
    ap.add_argument("--pencil-order", default="xyz", help="pencil 을 나누는 축 순서 (기본 xyz, 예: zyx)")
    ap.add_argument("--strong-ref", type=int, default=0,
                    help="strong 도메인 = 6×6×1 × n/REF (셀 크기를 weak_REF 와 같게; 기본 0 = 6×6×1 고정)")
    ap.add_argument("--slab-mesh", default="slab", choices=["slab", "pencil"],
                    help="weak slab 의 전역 격자: slab = z 로만 늘림(기본), pencil = 같은 np 의 pencil 과 같은 전역 격자 (slab/pencil 비교용)")
    ap.add_argument("--min-local", type=int, default=MIN_LOCAL,
                    help=f"랭크당 축 셀 수 하한 (기본 {MIN_LOCAL}; strong slab 의 큰 np 에 4 등으로 낮출 수 있다)")
    ap.add_argument("--out", default=os.path.join(HERE, "decks"),
                    help="덱 최상위 폴더 (기본 decks/; 다르면 job_scal.sh 에 DECKS= 로 넘긴다)")
    a = ap.parse_args()
    if sorted(a.pencil_order) != ["x", "y", "z"]:
        raise SystemExit(f"--pencil-order 는 x·y·z 를 한 번씩 (받은 값 {a.pencil_order})")

    n_written = 0
    for n in a.sizes:
        for mode in a.modes:
            for decomp in a.decomps:
                study = f"{mode}_{n}_{decomp}"
                for np_ in a.nps:
                    built, why = case(mode, n, decomp, np_, a.pencil_order, a.strong_ref, a.slab_mesh, a.min_local)
                    if built is None:
                        print(f"  skip {study} np {np_}: {why}")
                        continue
                    mesh, length, procs = built
                    title = (f"rbc_nob 확장성 {mode} — 전역 {mesh[0]}x{mesh[1]}x{mesh[2]}, "
                             f"{procs[0]}x{procs[1]}x{procs[2]} 랭크 ({decomp})")
                    text = deck_text(title, mesh, length, procs,
                                     f"max_step = {a.steps}",
                                     f"monitor_interval = {a.monitor}",
                                     f"\n[timing]\nwarmup_steps = {a.warmup}\n"
                                     + ("solver_phases = true\n" if a.solver_phases else ""))
                    write(os.path.join(a.out, study, f"{study}_np{np_}.ini"), text)
                    n_written += 1

    # 게이트 짝: 작은 slab 격자(64×64×16·np)를 60 스텝 — 노드 간 통신까지 거친다.
    for np_ in a.nps:
        mesh, length, procs = (64, 64, 16 * np_), BASE_L, (1, 1, np_)
        title = f"게이트 — 이 앱 (rbc_nob 대조), 64x64x{mesh[2]}, npz {np_}"
        write(os.path.join(a.out, "gate", f"gate_np{np_}.ini"),
              deck_text(title, mesh, length, procs, "max_step = 60", "monitor_interval = 10",
                        "\n[timing]\nwarmup_steps = 10\n"))
        title = f"게이트 — rbc_nob 대조 런 (통계·체크포인트 끔), 64x64x{mesh[2]}, npz {np_}"
        write(os.path.join(a.out, "gate", f"gate_ref_np{np_}.ini"),
              deck_text(title, mesh, length, procs, "end_time = 1.0e9\nmax_step = 60",
                        'monitor_interval = 10\nstats_start_time = 1.0e9\ncheckpoint_interval = 0\n'
                        'stats_dir = "../../log/gate_stats/"'))
        n_written += 2
    print(f"{n_written} decks → {a.out}")


if __name__ == "__main__":
    main()
