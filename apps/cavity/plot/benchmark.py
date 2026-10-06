#!/usr/bin/env python3
"""de Vahl Davis (1983) Table V 전 항목을 cavity 장 덤프에서 계산해 비교한다.

사용: python3 benchmark.py stats/1e5/cavity_field_00030062.dat [...]

항목 (논문 정의, p.252):
  |psi|_mid        공동 중점의 유동함수
  |psi|_max (x,z)  유동함수 최대 절대값과 위치 (1e5·1e6 에서만 중점이 아님)
  u_max (z)        연직 중앙면 x=1/2 의 최대 수평속도와 위치
  w_max (x)        수평 중앙면 z=1/2 의 최대 연직속도와 위치
  Nu_bar           공동 전체 평균 Nusselt  = ∬ (uT − ∂T/∂x) dx dz
  Nu_1/2           연직 중앙면 평균       = ∫ (uT − ∂T/∂x)|_{x=1/2} dz
  Nu_0             hot 벽 평균            = ∫ (−∂T/∂x)|_{x=0} dz  (2차 편측차분)
  Nu_max (z), Nu_min (z)   hot 벽 국소 Nu 의 극값과 위치
스케일: 논문은 L, κ, L²/κ → 속도 u·L/α = (free-fall u)·√(Ra·Pr); 유동함수 ψ/α 도 같은 배율.
극값·위치는 논문처럼 격자값에서 국소 4차 다항식(5점)으로 보간한다 (numpy 만 사용).
벽 Nu 극값은 논문도 "수치 미분으로 위치 결정" — 여기선 셀중심 열에 같은 보간.
"""
import sys
import numpy as np

BENCH = {  # Table V
    1e3: dict(psi_mid=1.174, psi_max=None, u_max=(3.649, 0.813), w_max=(3.697, 0.178),
              nu_bar=1.118, nu_half=1.118, nu_0=1.117, nu_max=(1.505, 0.092), nu_min=(0.692, 1.0)),
    1e4: dict(psi_mid=5.071, psi_max=None, u_max=(16.178, 0.823), w_max=(19.617, 0.119),
              nu_bar=2.243, nu_half=2.243, nu_0=2.238, nu_max=(3.528, 0.143), nu_min=(0.586, 1.0)),
    1e5: dict(psi_mid=9.111, psi_max=(9.612, 0.285, 0.601), u_max=(34.73, 0.855), w_max=(68.59, 0.066),
              nu_bar=4.519, nu_half=4.519, nu_0=4.509, nu_max=(7.717, 0.081), nu_min=(0.729, 1.0)),
    1e6: dict(psi_mid=16.32, psi_max=(16.750, 0.151, 0.547), u_max=(64.63, 0.850), w_max=(219.36, 0.0379),
              nu_bar=8.800, nu_half=8.799, nu_0=8.817, nu_max=(17.925, 0.0378), nu_min=(0.989, 1.0)),
}


def read_dump(path, swap_xz=False):
    head = open(path).readline()
    meta = dict(kv.split("=") for kv in head.replace("#", "").split()[3:])
    nx, nz = int(meta["nx"]), int(meta["nz"])
    ra, pr = float(meta["Ra"]), float(meta["Pr"])
    data = np.loadtxt(path, comments="#")
    n_u, n_w, n_t = (nx + 1) * nz, nx * (nz + 1), nx * nz
    U = data[:n_u, 2].reshape(nx + 1, nz)
    W = data[n_u:n_u + n_w, 2].reshape(nx, nz + 1)
    T = data[n_u + n_w:, 2].reshape(nx, nz)
    xf = data[:n_u, 0].reshape(nx + 1, nz)[:, 0]
    zc = data[:n_u, 1].reshape(nx + 1, nz)[0, :]
    xc = data[n_u:n_u + n_w, 0].reshape(nx, nz + 1)[:, 0]
    zf = data[n_u:n_u + n_w, 1].reshape(nx, nz + 1)[0, :]
    scale = np.sqrt(ra * pr)
    if swap_xz:
        # 축 교대 런(중력 x, hot/cold 가 z 벽): (x,z,U,W) → (z,x,W,U) 로 바꿔 논문 좌표계로
        U, W = W.T, U.T          # U' (nz+1 x nx) = 옛 W 전치, W' (nz x nx+1) = 옛 U 전치
        T = T.T
        xf, zc, xc, zf = zf, xc, zc, xf
        nx, nz = nz, nx
    return dict(nx=nx, nz=nz, ra=ra, pr=pr, U=U * scale, W=W * scale, T=T, xf=xf, zc=zc, xc=xc,
                zf=zf, t=float(meta["t"]))


def extremum(coord, values, sign=+1):
    """격자 극값 주변 5점에 4차 다항식을 맞춰(논문 방식) sign·values 의 최대값과 위치."""
    f = sign * np.asarray(values)
    k = int(np.argmax(f))
    lo, hi = max(0, k - 2), min(len(f), k + 3)
    if hi - lo < 5:  # 끝에 걸리면 창을 안쪽으로
        lo, hi = (0, 5) if lo == 0 else (len(f) - 5, len(f))
    c = np.polyfit(coord[lo:hi], f[lo:hi], 4)
    fine = np.linspace(coord[lo], coord[hi - 1], 4001)
    v = np.polyval(c, fine)
    j = int(np.argmax(v))
    return sign * v[j], fine[j]


def end_value(coord, values, at):
    """경계 z=at 의 값: 가장 가까운 4점 3차 다항식 외삽 (셀중심 → 벽)."""
    idx = np.argsort(np.abs(coord - at))[:4]
    c = np.polyfit(coord[idx], values[idx], 3)
    return float(np.polyval(c, at))


def analyze(d):
    nx, nz = d["nx"], d["nz"]
    U, W, T, xf, zc, xc, zf = d["U"], d["W"], d["T"], d["xf"], d["zc"], d["xc"], d["zf"]
    dx, dz = xf[1] - xf[0], zf[1] - zf[0]
    out = {}

    # ── 유동함수: 노드 (x 면 i, z 면 k) 에서 ψ = ∫₀^z u dz' (u 는 x 면·z 중심) ──
    psi = np.zeros((nx + 1, nz + 1))
    psi[:, 1:] = np.cumsum(U, axis=1) * dz
    out["psi_top_residual"] = float(np.max(np.abs(psi[:, -1])))  # 상단벽에서 0 이어야 (연속성)
    out["psi_mid"] = abs(float(psi[nx // 2, nz // 2]))
    # |ψ|_max: 격자 최대 근방 2D 2차 보간 (논문은 4차 다항식)
    a = np.abs(psi)
    i, k = np.unravel_index(np.argmax(a), a.shape)
    i = min(max(i, 1), nx - 1); k = min(max(k, 1), nz - 1)
    # 2D 포물선 피팅 (3x3)
    def parab(f, h):
        d1 = (f[2] - f[0]) / (2 * h); d2 = (f[2] - 2 * f[1] + f[0]) / h**2
        return (-d1 / d2 if d2 != 0 else 0.0), f[1] - 0.5 * d1**2 / d2 if d2 != 0 else f[1]
    sx, vx = parab(a[i - 1:i + 2, k], dx)
    sz, vz = parab(a[i, k - 1:k + 2], dz)
    out["psi_max"] = (float(max(vx, vz)), float(xf[i] + sx), float(zf[k] + sz))

    # ── u_max on x = 1/2 (x 면 i = nx/2), w_max on z = 1/2 (z 면 k = nz/2) ──
    uline = U[nx // 2, :]
    v, z = extremum(zc, uline, +1 if abs(uline.max()) >= abs(uline.min()) else -1)
    out["u_max"] = (abs(float(v)), float(z))
    wline = W[:, nz // 2]
    v, x = extremum(xc, wline, +1 if abs(wline.max()) >= abs(wline.min()) else -1)
    out["w_max"] = (abs(float(v)), float(x))

    # ── 열유속 Q = uT − ∂T/∂x ──
    t_hot, t_cold = 0.5, -0.5
    # hot 벽 국소 Nu (2차 편측: 벽, dx/2, 3dx/2)
    nu_wall = -(-8 * t_hot + 9 * T[0, :] - T[1, :]) / (3 * dx)
    out["nu_0"] = float(np.sum(nu_wall) * dz)  # 셀 중점 적분 (균일 격자 midpoint rule)
    v, z = extremum(zc, nu_wall, +1); out["nu_max"] = (float(v), float(z))
    v, z = extremum(zc, nu_wall, -1); out["nu_min"] = (float(v), float(z))
    # 벽 끝점(z=0,1)의 Nu 는 셀중심 외삽: 논문 Nu_min 은 z=1 에서 — 스플라인 끝값으로 보고
    out["nu_wall_at_z1"] = end_value(zc, nu_wall, 1.0); out["nu_wall_at_z0"] = end_value(zc, nu_wall, 0.0)
    # x = 1/2 면: u 는 면값, T 면값 = 평균, ∂T/∂x 중앙차분
    i = nx // 2
    Tf = 0.5 * (T[i - 1, :] + T[i, :]); dTdx = (T[i, :] - T[i - 1, :]) / dx
    out["nu_half"] = float(np.sum(U[i, :] * Tf - dTdx) * dz)
    # 공동 평균: 셀중심 Q, ∂T/∂x 는 내부 중앙차분, 벽 셀은 2차 편측 (ghost 없이)
    uc = 0.5 * (U[:-1, :] + U[1:, :])
    dT = np.zeros_like(T)
    dT[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * dx)
    dT[0, :] = (T[1, :] - (2 * t_hot - T[0, :])) / (2 * dx)      # 거울 ghost 중앙차분
    dT[-1, :] = ((2 * t_cold - T[-1, :]) - T[-2, :]) / (2 * dx)
    out["nu_bar"] = float(np.sum(uc * T - dT) * dx * dz)
    return out


def report(path, swap_xz=False):
    d = read_dump(path, swap_xz)
    r = analyze(d)
    b = BENCH[d["ra"]]
    print(f"\n=== {path}  Ra={d['ra']:g} {d['nx']}x{d['nz']} t={d['t']:.1f}  (psi top residual {r['psi_top_residual']:.2e})")
    print(f"{'quantity':<14}{'ours':>12}{'bench':>12}{'err%':>8}   location (ours / bench)")
    def row(name, ours, bench, loc_ours="", loc_bench=""):
        err = 100 * (ours / bench - 1) if bench else float("nan")
        print(f"{name:<14}{ours:12.4f}{bench:12.4f}{err:8.2f}   {loc_ours} / {loc_bench}")
    row("|psi|_mid", r["psi_mid"], b["psi_mid"])
    if b["psi_max"]:
        row("|psi|_max", r["psi_max"][0], b["psi_max"][0], f"x={r['psi_max'][1]:.3f} z={r['psi_max'][2]:.3f}", f"x={b['psi_max'][1]} z={b['psi_max'][2]}")
    else:
        print(f"{'|psi|_max':<14}{r['psi_max'][0]:12.4f}{'(=mid)':>12}           x={r['psi_max'][1]:.3f} z={r['psi_max'][2]:.3f}")
    zu = r["u_max"][1]; zu_sym = zu if abs(zu - b["u_max"][1]) < abs(1 - zu - b["u_max"][1]) else 1 - zu
    row("u_max", r["u_max"][0], b["u_max"][0], f"z={zu_sym:.3f}", f"z={b['u_max'][1]}")
    xw = r["w_max"][1]; xw_sym = xw if abs(xw - b["w_max"][1]) < abs(1 - xw - b["w_max"][1]) else 1 - xw
    row("w_max", r["w_max"][0], b["w_max"][0], f"x={xw_sym:.4f}", f"x={b['w_max'][1]}")
    row("Nu_bar", r["nu_bar"], b["nu_bar"])
    row("Nu_1/2", r["nu_half"], b["nu_half"])
    row("Nu_0", r["nu_0"], b["nu_0"])
    row("Nu_max", r["nu_max"][0], b["nu_max"][0], f"z={r['nu_max'][1]:.4f}", f"z={b['nu_max'][1]}")
    row("Nu_min", r["nu_min"][0], b["nu_min"][0], f"z={r['nu_min'][1]:.3f} (extrap z=1: {r['nu_wall_at_z1']:.3f})", f"z={b['nu_min'][1]}")
    return r


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--swap-xz"]
    for p in args:
        report(p, swap_xz="--swap-xz" in sys.argv)
