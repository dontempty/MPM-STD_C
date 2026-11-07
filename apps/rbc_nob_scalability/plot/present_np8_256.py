# 발표용 그림 C — weak 512³/GPU pencil zyx [222 224 244 444 448 488] = np 8..256, 4 ranks/node, 평면 경로.
#   C1 step 시간, C2 효율, C3 scaled speedup, C4 압력 시간 (참고선: 4 회 전치 np 8–64), C5 step 분해,
#   C6 압력 내부 분해, C7 np 최대 손실 순위, C8 이벤트별 small multiples, C9 이벤트별 효율
# 사용 (apps/rbc_nob_scalability/ 에서):  python3 plot/present_np8_256.py [RES_PLAIN RES_PHASES]
#   기본 results/weak512_zyx_plane256 (REPS 2) 와 results/weak512_zyx_plane256_phases (solver_phases, REPS 1)
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pathlib
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
APP = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "plot"))
import scaling_plot as sp
from present_np1_64 import pick, pressure_groups, pressure_bars   # A·B 그림과 같은 묶음 정의

st = "weak_512_pencil"
BEFORE = "results/weak512_zyx_selfcopy"            # 4 회 전치 (np 8–64 만) — 참고선
DECOMP = {8: "2x2x2", 16: "2x2x4", 32: "2x4x4", 64: "4x4x4", 128: "4x4x8", 256: "4x8x8"}


def event_efficiency(d, fname):
    """이벤트별 효율 E_e(np) = T_e(np0) / T_e(np). 큰 구간(T(np0) ≥ 5 ms)과 작은 구간을 나눠 그린다
       (작은 구간은 ms 단위 잡음이 E 를 크게 흔든다). monitor 는 제외 (50 스텝마다 한 번)."""
    nps = list(d)
    rows = [(e, d[nps[0]]["phases"].get(e, 0.0)) for e in sp.EVENTS if e != "monitor"]
    big = [(e, t0) for e, t0 in rows if t0 >= 5.0]
    small = [(e, t0) for e, t0 in rows if 0 < t0 < 5.0]
    palette = list(sp.SLOTS) + ["#7f3c8d", "#11a579", "#e68310", "#3969ac", "#80ba5a", "#a5aa99"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=130, sharey=True)
    for ax, group, name in ((axes[0], big, "large events  (T(%d) ≥ 5 ms)" % nps[0]),
                            (axes[1], small, "small events  (T(%d) < 5 ms)" % nps[0])):
        ax.axhline(1.0, color=sp.GUIDE, lw=1.1, zorder=1)
        for i, (e, t0) in enumerate(sorted(group, key=lambda r: -r[1])):
            ys = [t0 / d[n]["phases"][e] if d[n]["phases"].get(e, 0) > 0 else float("nan") for n in nps]
            ax.plot(nps, ys, lw=2, marker="o", ms=5, color=palette[i % len(palette)], mec=sp.SURFACE, mew=1.0,
                    zorder=3, label="%s  (T(%d) %.1f ms, E(%d) %.2f)" % (e, nps[0], t0, nps[-1], ys[-1]))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(nps))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["%d" % n for n in nps])
        ax.set_xlim(nps[0] / 1.3, nps[-1] * 1.3)
        ax.set_ylim(0, 1.3)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("GPUs  np")
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    axes[0].set_ylabel("event efficiency  E_e = T_e(%d) / T_e(np)" % nps[0])
    fig.suptitle("Weak np %d–%d — efficiency per event, plane path (ms/step, rank max; 4 ranks/node)"
                 % (nps[0], nps[-1]), x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    sp._save(fig, fname)


def main():
    os.chdir(APP)
    sp.FIGDIR = APP / "plot" / "figures" / "present"
    plane = sys.argv[1] if len(sys.argv) > 1 else "results/weak512_zyx_plane256"
    ph_plane = sys.argv[2] if len(sys.argv) > 2 else "results/weak512_zyx_plane256_phases"

    c = {st + "@p": pick(plane, 8, 256), st + "@b": pick(BEFORE, 8, 64)}
    lab = {st + "@p": "plane path (2 transposes)", st + "@b": "before (4 transposes, np 8–64)"}
    print("== np 8-256"); sp.table(c)
    nps = sorted(c[st + "@p"])
    top = max(r["total"] for d in c.values() for r in d.values())
    sp.line_chart(c, "total", title="Weak np 8–256 — time per step (512³/GPU, pencil zyx, 4 ranks/node)",
                  ylabel="time per step [ms]", logy=False, ylim=(0, top * 1.2), labels=lab,
                  ideal=lambda x, d: d[min(d)]["total"], ideal_label="ideal  flat (np = 8)", fmt="%.1f ms",
                  legend_loc="lower left", fname="C1_np8-256_time")
    sp.line_chart(c, "eff", title="Weak np 8–256 — efficiency  E = T(8) / T(np)", ylabel="efficiency (base np 8)",
                  logy=False, labels=lab, ideal=lambda x, d: 1.0, ideal_label="ideal  1.0", ylim=(0, 1.15),
                  fmt="%.2f", legend_loc="lower left", fname="C2_np8-256_efficiency")
    sp.line_chart(c, "speedup", title="Weak np 8–256 — scaled speedup  S = np·T(8) / (8·T(np))",
                  ylabel="scaled speedup (base np 8)", logy=True, log_base=2, labels=lab,
                  ideal=lambda x, d: x / min(d), ideal_label="ideal  np / 8", fmt="%.2fx", legend_loc="upper left",
                  fname="C3_np8-256_speedup")
    ptop = max(r["pressure"] for d in c.values() for r in d.values())
    sp.line_chart(c, "pressure", title="Weak np 8–256 — pressure_solve per step", ylabel="pressure_solve [ms/step]",
                  logy=False, ylim=(0, ptop * 1.2), labels=lab, fmt="%.1f ms", legend_loc="upper left",
                  fname="C4_np8-256_pressure")
    php = {n: r for n, r in sp.collect([ph_plane], st).items() if n >= 8}
    if php:
        sp.phase_chart(php, title="Weak np 8–256 — per-step breakdown, plane path", fname="C5_np8-256_phases_plane")
        pressure_bars([("plane", php)], sorted(php), "Weak np 8–256 — inside pressure_solve, plane path",
                      "C6_np8-256_pressure_breakdown")
    d = c[st + "@p"]
    sp.loss_chart(d, "weak", title="Weak — loss by event at np = %d vs np 8 (plane path)" % nps[-1],
                  fname="C7_np8-256_loss_plane")
    sp.event_grid(d, "weak", title="Weak np 8–256 — per-event scalability, plane path", cols=4,
                  fname="C8_np8-256_event_grid")
    event_efficiency(d, "C9_np8-256_event_efficiency")

    print("\n== plane path, plain runs")
    print("%4s %7s %9s %9s %6s" % ("np", "decomp", "ms/step", "pressure", "E"))
    for n in nps:
        r = d[n]
        print("%4d %7s %9.1f %9.1f %6.3f" % (n, DECOMP.get(n, "?"), r["total"], r["pressure"], r["eff"]))
    print("\n== per-event loss at np %d (vs np %d)" % (nps[-1], nps[0]))
    print("%-18s %8s %8s %7s %6s" % ("event", "T(%d)" % nps[0], "T(%d)" % nps[-1], "loss", "E"))
    for e, loss, eff, t0, t in sp.event_loss(d, "weak"):
        print("%-18s %8.2f %8.2f %7.2f %6.2f" % (e, t0, t, loss, eff))
    if php:
        print("\n== plane path, pressure internals (phases runs)")
        for n in sorted(php):
            print(n, " ".join("%s %.1f" % (g, v) for g, v in pressure_groups(php[n]["phases"])),
                  " | tdma_comm_pressure_z %.1f" % php[n]["phases"].get("tdma_comm_pressure_z", 0))


if __name__ == "__main__":
    main()
