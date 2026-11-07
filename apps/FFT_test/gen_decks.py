#!/usr/bin/env python3
"""FFT_test 입력 덱 생성 — decks/<study>/<study>_np<N>.ini (plan.md §5·§7).

  test          작은 격자 6 종, 4 랭크. 분할·불균등 블록·슬랩·균일 z 를 모두 건드린다 (순환 격자 q = npx·npy, q² | ny).
  weak_512_v2   GPU 당 512³, np 1~64. 격자·프로세스 격자는 apps/rbc_nob_scalability/decks_zyx/weak_512_pencil 과 같다.
                v1 측정(results/weak_512)과 섞이지 않게 이름을 나눴다.
  compare_weak256, compare_strong1024
                CaNS·PaScaL 과 같은 문제 (COMPARISON_CONTRACT_HANGANG.md): 균일 격자, x·y 주기, z Neumann,
                프로세스 격자 1 x np2 x np3, GPU 64 장까지. RMS 대조로 같은 이산 문제임을 확인한다.
  transport_weak256, transport_strong1024
                위와 같은 문제에서 MPM-STD 펜슬 경로만 잰다 (전치 통신 방식 비교). 전치가 있는 np 4~64.
  cube_weak512  weak_512_v2 와 같은 격자·프로세스 격자·z 스트레칭 (cube 분할), np 4~256. MPM-STD 평면묶음·펜슬 경로의 통신 방식 비교.
                해석 문제가 주기 조건을 만족하도록 길이는 격자에 비례 (2 x 격자 / 512). 스트레칭 z 는 방식 간 RMS 로 대조한다.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = dict(lx=1.3, ly=0.9, lz=1.0)
GAMMA = 2.6


def deck(nx, ny, nz, npx, npy, npz, uniform=False, warmup=5, solves=20, tolerance=1e-9, lengths=None, fft=True,
         accuracy=True, contract=False, only=""):
    z = "uniform_z = true" if uniform else f"uniform_z = false\ngamma_z = {GAMMA}"
    L = lengths or globals()["L"]
    flag = lambda v: "true" if v else "false"
    return f"""[mesh]
nx = {nx}
ny = {ny}
nz = {nz}

[domain]
lx = {L['lx']}
ly = {L['ly']}
lz = {L['lz']}

[stretch]
{z}

[process]
npx = {npx}
npy = {npy}
npz = {npz}

[check]
fft = {flag(fft)}
accuracy = {flag(accuracy)}
contract = {flag(contract)}
tolerance = {tolerance}

[bench]
warmup = {warmup}
solves = {solves}
phases = true
""" + (f"only = {only}\n" if only else "")


def write(study, name, text):
    folder = os.path.join(HERE, "decks", study)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}.ini"), "w") as f:
        f.write(text)


TESTS = {
    "t1_2x2x1": dict(nx=64, ny=64, nz=32, npx=2, npy=2, npz=1),
    "t2_slab_1x1x4": dict(nx=64, ny=64, nz=32, npx=1, npy=1, npz=4),
    "t3_uneven_x_4x1x1": dict(nx=30, ny=64, nz=26, npx=4, npy=1, npz=1),
    "t4_uneven_z_2x1x2": dict(nx=64, ny=32, nz=31, npx=2, npy=1, npz=2),   # z 16 + 15
    "t5_1x4x1": dict(nx=64, ny=64, nz=32, npx=1, npy=4, npz=1),
    "t6_uniform_z_2x2x1": dict(nx=64, ny=64, nz=32, npx=2, npy=2, npz=1, uniform=True),
}
for name, a in TESTS.items():
    write("test", name, deck(**a, warmup=2, solves=5))

WEAK_512 = {
    1: (512, 512, 512, 1, 1, 1),
    2: (512, 512, 1024, 1, 1, 2),
    4: (512, 1024, 1024, 1, 2, 2),
    8: (1024, 1024, 1024, 2, 2, 2),
    16: (1024, 1024, 2048, 2, 2, 4),
    32: (1024, 2048, 2048, 2, 4, 4),
    64: (2048, 2048, 2048, 4, 4, 4),
}
for np, (nx, ny, nz, px, py, pz) in WEAK_512.items():
    assert ny % (px * py) ** 2 == 0, f"np {np}: q^2 does not divide ny"
    write("weak_512_v2", f"weak_512_v2_np{np}", deck(nx, ny, nz, px, py, pz))
# CaNS·PaScaL 비교 — tools/generate_case.py 의 GRID_TABLE, strong: 전역 고정, weak: x 고정 + y·z 로 늘림
GRID_TABLE = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4), 32: (4, 8), 64: (8, 8)}
for np, (np2, np3) in GRID_TABLE.items():
    nx, ny, nz = 256, 256 * np2, 256 * np3
    lengths = dict(lx=2.0, ly=2.0 * np2, lz=2.0 * np3)
    assert ny % np2 ** 2 == 0
    write("compare_weak256", f"compare_weak256_np{np}",
          deck(nx, ny, nz, 1, np2, np3, uniform=True, lengths=lengths, fft=False, contract=True))
    if np == 1:
        continue   # 1024³ 은 GPU 1 장 메모리에 안 들어간다 (계약도 np ≥ 2)
    assert 1024 % np2 ** 2 == 0
    # 난수 rhs 대조는 호스트 루프라 랭크당 셀이 많으면 오래 걸린다. np 2 는 해석해 RMS 만 본다.
    write("compare_strong1024", f"compare_strong1024_np{np}",
          deck(1024, 1024, 1024, 1, np2, np3, uniform=True, lengths=dict(lx=2.0, ly=2.0, lz=2.0), fft=False,
               accuracy=np >= 4, contract=True))
# 전치 통신 방식 비교 — 전치가 실제로 있는 np (np2 > 1) 만, MPM-STD 펜슬 경로만
for np, (np2, np3) in GRID_TABLE.items():
    if np2 == 1:
        continue
    write("transport_weak256", f"transport_weak256_np{np}",
          deck(256, 256 * np2, 256 * np3, 1, np2, np3, uniform=True, lengths=dict(lx=2.0, ly=2.0 * np2, lz=2.0 * np3),
               fft=False, accuracy=False, contract=True, only="mpm_pencil"))
    write("transport_strong1024", f"transport_strong1024_np{np}",
          deck(1024, 1024, 1024, 1, np2, np3, uniform=True, lengths=dict(lx=2.0, ly=2.0, lz=2.0), fft=False,
               accuracy=False, contract=True, only="mpm_pencil"))
# cube 분할의 통신 방식 비교 — 전치가 있는 np 4 이상. np 128·256 은 rbc_nob_scalability/decks_zyx/weak_512_pencil 과 같은 격자
CUBE_LARGE = {128: (2048, 2048, 4096, 4, 4, 8), 256: (2048, 4096, 4096, 4, 8, 8)}
for np, (nx, ny, nz, px, py, pz) in {**WEAK_512, **CUBE_LARGE}.items():
    if px * py == 1:
        continue
    assert ny % (px * py) ** 2 == 0, f"np {np}: q^2 does not divide ny"
    write("cube_weak512", f"cube_weak512_np{np}",
          deck(nx, ny, nz, px, py, pz, lengths=dict(lx=2.0 * nx / 512, ly=2.0 * ny / 512, lz=2.0 * nz / 512), fft=False,
               accuracy=False, contract=True, only="mpm_plane,mpm_pencil"))
print("decks written under", os.path.join(HERE, "decks"))
