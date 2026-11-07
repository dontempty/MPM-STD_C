# 발표용 그림 A·B — plot/figures/present/ (압력 솔버 최적화 전후, weak 512³/GPU pencil zyx, 4 ranks/node)
#   A1–A5: np 1–8 (기준 np 1)   B1–B7: np 8–64 (기준 np 8)
# 사용 (apps/rbc_nob_scalability/ 에서):  python3 plot/present_np1_64.py
# 결과 폴더: results/weak512_zyx_selfcopy (4 회 전치, 인덱스 팩), results/weak512_zyx_packopt (4 회 전치, 구조화 팩),
#           results/weak512_zyx_plane (평면 경로) 와 각각의 *_phases (solver_phases = true 런)
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pathlib
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
APP = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "plot"))
import scaling_plot as sp

st = "weak_512_pencil"
BEFORE, PACK, PLANE = "results/weak512_zyx_selfcopy", "results/weak512_zyx_packopt", "results/weak512_zyx_plane"
PH_BEFORE, PH_PACK, PH_PLANE = ("results/weak512_zyx_phases", "results/weak512_zyx_packopt_phases",
                                "results/weak512_zyx_plane_phases")


def pick(res, lo, hi):
    d = {n: r for n, r in sp.collect([res], st).items() if lo <= n <= hi}
    for r in d.values():
        r["pressure"] = r["phases"].get("pressure_solve", 0.0)
    return sp.metrics(dict(sorted(d.items())), "weak")


def pressure_groups(ph):
    """압력 솔버 안을 묶음으로: 전치 MPI / 전치 팩·언팩 / FFT·재배열 / z TDMA / 기타."""
    comm = sum(v for k, v in ph.items() if k.startswith("pressure.comm."))
    trans = sum(v for k, v in ph.items() if k.startswith("pressure.pencil.") and k.split(".")[-1] in
                ("block_to_lines", "x_to_y", "y_to_x", "lines_to_block")) + \
        sum(v for k, v in ph.items() if k.startswith("pressure.plane."))
    fft = sum(ph.get(k, 0.0) for k in ("pressure.fft_x_fwd", "pressure.fft_y_fwd", "pressure.fft_y_bwd",
                                       "pressure.fft_x_bwd", "pressure.reorder_x_to_y", "pressure.reorder_y_to_x",
                                       "pressure.pencil.fft_x_fwd", "pressure.pencil.fft_y_fwd",
                                       "pressure.pencil.fft_y_bwd", "pressure.pencil.fft_x_bwd",
                                       "pressure.pack_lines", "pressure.scatter_lines"))
    tdma = ph.get("pressure.tdma_z", 0.0) + ph.get("pressure.build_z", 0.0)
    packs = max(trans - comm, 0.0)
    # 구간마다 랭크 max 라 묶음 합 ≠ pressure_solve(랭크 max) — 나머지로 구하지 않고 직접 더한다
    other = ph.get("pressure.mean_remove", 0.0) + ph.get("pressure.halo", 0.0) + ph.get("pressure.pack_bands", 0.0)
    return [("transpose MPI", comm), ("transpose pack/unpack", packs), ("FFT + reorder", fft),
            ("z TDMA (build + solve)", tdma), ("other (mean, halo)", other)]


def pressure_bars(sets, nps, title, fname):
    """np 마다 버전별 막대(누적: 압력 솔버 내부 묶음)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=130)
    colors = list(sp.SLOTS[:4]) + [sp.NEUTRAL]
    w = 0.8 / len(sets)
    names = None
    for si, (label, d) in enumerate(sets):
        for xi, n in enumerate(nps):
            if n not in d:
                continue
            groups = pressure_groups(d[n]["phases"])
            names = [g for g, _ in groups]
            bottom = 0.0
            x = xi + (si - (len(sets) - 1) / 2) * w
            for gi, (_, v) in enumerate(groups):
                ax.bar(x, v, width=w * 0.9, bottom=bottom, color=colors[gi], edgecolor=sp.SURFACE, linewidth=1.0,
                       zorder=3)
                bottom += v
            ax.annotate("%.0f" % d[n]["phases"]["pressure_solve"], xy=(x, bottom), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=sp.INK)
            ax.annotate(label, xy=(x, 0), xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                        fontsize=7, color=sp.INK2, rotation=90)
    ax.set_xticks(range(len(nps)))
    ax.set_xticklabels(["\n\n\n\n\nnp %d" % n for n in nps])
    ax.set_ylabel("pressure_solve [ms/step]  (each part: rank max)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=c, label=nm) for c, nm in zip(colors, names)][::-1], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.tight_layout()
    sp._save(fig, fname)


def main():
    os.chdir(APP)
    sp.FIGDIR = APP / "plot" / "figures" / "present"
    # ── A. np 1–8 (기준 np 1) ─────────────────────────────────────────────────
    a = {st + "@b": pick(BEFORE, 1, 8), st + "@k": pick(PACK, 1, 8), st + "@p": pick(PLANE, 1, 8)}
    lab = {st + "@b": "before (4 transposes, index pack)", st + "@k": "4 transposes, structured pack",
           st + "@p": "plane path (2 transposes)"}
    print("== np 1-8"); sp.table(a)
    top = max(r["total"] for d in a.values() for r in d.values())
    sp.line_chart(a, "total", title="Weak np 1–8 — time per step (512³/GPU, pencil zyx, 4 ranks/node)",
                  ylabel="time per step [ms]", logy=False, ylim=(0, top * 1.2), labels=lab,
                  ideal=lambda x, d: d[min(d)]["total"], ideal_label="ideal  flat (np = 1)", fmt="%.1f ms",
                  legend_loc="lower left", fname="A1_np1-8_time")
    sp.line_chart(a, "eff", title="Weak np 1–8 — efficiency  E = T(1) / T(np)", ylabel="efficiency",
                  logy=False, labels=lab, ideal=lambda x, d: 1.0, ideal_label="ideal  1.0", ylim=(0, 1.15),
                  fmt="%.2f", legend_loc="lower left", fname="A2_np1-8_efficiency")
    ptop = max(r["pressure"] for d in a.values() for r in d.values())
    sp.line_chart(a, "pressure", title="Weak np 1–8 — pressure_solve per step", ylabel="pressure_solve [ms/step]",
                  logy=False, ylim=(0, ptop * 1.2), labels=lab, fmt="%.1f ms", legend_loc="upper left",
                  fname="A3_np1-8_pressure")
    pa = [("before", sp.collect([PH_BEFORE], st)), ("pack opt", sp.collect([PH_PACK], st)),
          ("plane", sp.collect([PH_PLANE], st))]
    pressure_bars(pa, [1, 2, 4, 8], "Weak np 1–8 — inside pressure_solve", "A4_np1-8_pressure_breakdown")
    sp.phase_chart(pa[2][1], title="Weak np 1–8 — per-step breakdown, plane path", fname="A5_np1-8_phases_plane")

    # ── B. np 8–64 (기준 np 8) ────────────────────────────────────────────────
    b = {st + "@b": pick(BEFORE, 8, 64), st + "@p": pick(PLANE, 8, 64)}
    lab_b = {st + "@b": "before (4 transposes, index pack)", st + "@p": "plane path (2 transposes)"}
    print("\n== np 8-64"); sp.table(b)
    top = max(r["total"] for d in b.values() for r in d.values())
    sp.line_chart(b, "total", title="Weak np 8–64 — time per step (512³/GPU, pencil zyx, 4 ranks/node)",
                  ylabel="time per step [ms]", logy=False, ylim=(0, top * 1.2), labels=lab_b,
                  ideal=lambda x, d: d[min(d)]["total"], ideal_label="ideal  flat (np = 8)", fmt="%.1f ms",
                  legend_loc="lower left", fname="B1_np8-64_time")
    sp.line_chart(b, "eff", title="Weak np 8–64 — efficiency  E = T(8) / T(np)", ylabel="efficiency (base np 8)",
                  logy=False, labels=lab_b, ideal=lambda x, d: 1.0, ideal_label="ideal  1.0", ylim=(0, 1.15),
                  fmt="%.2f", legend_loc="lower left", fname="B2_np8-64_efficiency")
    sp.line_chart(b, "speedup", title="Weak np 8–64 — scaled speedup  S = np·T(8) / (8·T(np))",
                  ylabel="scaled speedup (base np 8)", logy=True, log_base=2, labels=lab_b,
                  ideal=lambda x, d: x / min(d), ideal_label="ideal  np / 8", fmt="%.2fx", legend_loc="upper left",
                  fname="B3_np8-64_speedup")
    ptop = max(r["pressure"] for d in b.values() for r in d.values())
    sp.line_chart(b, "pressure", title="Weak np 8–64 — pressure_solve per step", ylabel="pressure_solve [ms/step]",
                  logy=False, ylim=(0, ptop * 1.2), labels=lab_b, fmt="%.1f ms", legend_loc="upper left",
                  fname="B4_np8-64_pressure")
    php = {n: r for n, r in sp.collect([PH_PLANE], st).items() if n >= 8}
    sp.phase_chart(php, title="Weak np 8–64 — per-step breakdown, plane path", fname="B5_np8-64_phases_plane")
    pressure_bars([("plane", php)], [8, 16, 32, 64], "Weak np 8–64 — inside pressure_solve, plane path",
                  "B6_np8-64_pressure_breakdown")
    sp.loss_chart(b[st + "@p"], "weak", title="Weak — loss by event at np = 64 vs np 8 (plane path)",
                  fname="B7_np8-64_loss_plane")

    for tag, d in (("np1-8 plane", pa[2][1]), ("np8-64 plane", php)):
        print("\n== %s pressure internals" % tag)
        for n in d:
            print(n, " ".join("%s %.2f" % (g, v) for g, v in pressure_groups(d[n]["phases"])),
                  " | tdma_comm_pressure_z %.2f" % d[n]["phases"].get("tdma_comm_pressure_z", 0))


if __name__ == "__main__":
    main()
