# 논문 그림 — pencil(cube) vs slab, weak 256³/512³ + strong 1024³/2048³ (results/paper_zyx, 2026-11-16 캠페인)
#   스타일 (사용자 규칙 2026-11-11): 축 라벨 20 pt, 그림 제목 없음 (설명은 캡션), 패널 태그 (a)(b) 만.
#   색은 구현에 고정: pencil = 파랑, slab = 주황 (모든 그림 동일). 3 회 반복의 중앙값 런을 쓰고 min–max 를 오차 막대로.
#   출력: plot/figures/paper/  F1 weak 시간, F2 weak 효율, F3 slab/pencil 비, F4 strong 시간, F5 strong 효율,
#        F6 압력 비중, F7 weak 256 압력 내부 (phases), paper_table.md
# 사용 (apps/rbc_nob_scalability/ 에서):  /scratch/t525a06/KDH/.venv-plot/bin/python plot/present_paper.py
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pathlib
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
APP = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "plot"))
import scaling_plot as sp
import summarize
from present_np1_64 import pressure_groups

RES, RES_PH = "results/paper_zyx", "results/paper_zyx_phases"
FIG = APP / "plot" / "figures" / "paper"
COL = {"pencil": "#2a78d6", "slab": "#eb6834"}
MK = {"pencil": "o", "slab": "s"}
LAB = {"pencil": "pencil (cube blocks)", "slab": "slab (1×1×np)"}
BASE = {"weak_256": 8, "weak_512": 8, "strong_1024": 8, "strong_2048": 32}
plt.rcParams.update({"axes.labelsize": 20, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def runs(res, study):
    """{np: {"med": 중앙값 런 phases(랭크 max), "all": [step_total...]}} — 3 회 반복 전부."""
    out = {}
    for (np_, nodes, rpn), lst in sp._runs([res], study).items():
        tot = sorted(r[2]["step_total"][2] for r in lst)
        lst = sorted(lst, key=lambda r: r[2]["step_total"][2])
        med = lst[(len(lst) - 1) // 2][2]
        out[np_] = {"med": {k: v[2] for k, v in med.items()}, "all": tot, "n": len(lst)}
    return dict(sorted(out.items()))


def load(res):
    d = {}
    for size in ("weak_256", "weak_512", "strong_1024", "strong_2048"):
        for dec in ("pencil", "slab"):
            try:
                d[(size, dec)] = runs(res, f"{size}_{dec}")
            except Exception:
                d[(size, dec)] = {}
    return d


def axis_np(ax, nps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(nps)
    ax.set_xticklabels([str(n) for n in nps])
    ax.set_xlabel("GPUs")
    ax.grid(True, axis="y", color=sp.GRID, lw=0.8)
    ax.set_axisbelow(True)


def tag(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top", ha="left")


def errbar(ax, d, dec, key=None, scale=None, **kw):
    nps = sorted(d)
    if not nps:
        return
    y = [d[n]["med"]["step_total"] if key is None else key(n, d[n]) for n in nps]
    lo = [y[i] - min(d[n]["all"]) for i, n in enumerate(nps)] if key is None else None
    hi = [max(d[n]["all"]) - y[i] for i, n in enumerate(nps)] if key is None else None
    ax.errorbar(nps, y, yerr=[lo, hi] if lo else None, color=COL[dec], marker=MK[dec], ms=8, lw=2, capsize=3,
                label=LAB[dec], zorder=3, **kw)
    return nps, y


def time_fig(d, sizes, fname, ylabel="time per step [ms]"):
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.2 * len(sizes), 5.0), dpi=150)
    for ax, size, t in zip(axes, sizes, "abcd"):
        allnp = set()
        for dec in ("pencil", "slab"):
            r = errbar(ax, d[(size, dec)], dec)
            if r:
                allnp |= set(r[0])
        if not allnp:
            ax.set_visible(False)
            continue
        axis_np(ax, sorted(allnp))
        top = max(max(dd[n]["all"]) for dec in ("pencil", "slab") for dd in [d[(size, dec)]] for n in dd)
        ax.set_ylim(0, top * 1.35)
        ax.set_ylabel(ylabel)
        n = size.split("_")[1]
        tag(ax, f"({t}) {size.split('_')[0]} {n}³" + ("/GPU" if size.startswith("weak") else ""))
        if size.startswith("weak"):
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False)
        else:
            ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


def eff_fig(d, sizes, fname):
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.2 * len(sizes), 5.0), dpi=150)
    for ax, size, t in zip(axes, sizes, "abcd"):
        b = BASE[size]
        allnp = set()
        for dec in ("pencil", "slab"):
            dd = d[(size, dec)]
            if b not in dd:
                continue
            t0 = dd[b]["med"]["step_total"]
            if size.startswith("weak"):
                key = lambda n, r: t0 / r["med"]["step_total"]
            else:
                key = lambda n, r: t0 * b / (r["med"]["step_total"] * n)
            res = errbar(ax, {n: r for n, r in dd.items() if n >= b}, dec, key=key)
            if res:
                allnp |= set(res[0])
        if not allnp:
            ax.set_visible(False)
            continue
        axis_np(ax, sorted(allnp))
        ax.axhline(1.0, color=sp.GUIDE, lw=1, ls="--")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("parallel efficiency")
        n = size.split("_")[1]
        tag(ax, f"({t}) {size.split('_')[0]} {n}³" + ("/GPU" if size.startswith("weak") else "") + f", base {b} GPUs")
        ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


def ratio_fig(d, fname):
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    styles = {"weak_256": ("-", "o"), "weak_512": ("-", "s"), "strong_1024": ("--", "^"), "strong_2048": ("--", "D")}
    allnp = set()
    for size, (ls, mk) in styles.items():
        p, s = d[(size, "pencil")], d[(size, "slab")]
        nps = sorted(set(p) & set(s))
        if not nps:
            continue
        y = [s[n]["med"]["step_total"] / p[n]["med"]["step_total"] for n in nps]
        label = size.replace("_", " ") + ("³/GPU" if size.startswith("weak") else "³")
        ax.plot(nps, y, ls=ls, marker=mk, ms=8, lw=2, color=COL["slab"] if size.startswith("weak") else COL["pencil"],
                alpha=1.0 if size.endswith(("256", "1024")) else 0.6, label=label, zorder=3)
        allnp |= set(nps)
    axis_np(ax, sorted(allnp))
    ax.axhline(1.0, color=sp.GUIDE, lw=1, ls="--")
    ax.set_ylabel("T(slab) / T(pencil)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


def share_fig(d, sizes, fname):
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.2 * len(sizes), 5.0), dpi=150)
    for ax, size, t in zip(axes, sizes, "abcd"):
        allnp = set()
        for dec in ("pencil", "slab"):
            dd = d[(size, dec)]
            res = errbar(ax, dd, dec, key=lambda n, r: r["med"].get("pressure_solve", 0.0) / r["med"]["step_total"])
            if res:
                allnp |= set(res[0])
        if not allnp:
            ax.set_visible(False)
            continue
        axis_np(ax, sorted(allnp))
        ax.set_ylim(0, 0.8)
        ax.set_ylabel("pressure solve / step")
        n = size.split("_")[1]
        tag(ax, f"({t}) {size.split('_')[0]} {n}³" + ("/GPU" if size.startswith("weak") else ""))
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


def comm_fig(d, sizes, fname):
    """통신 항목: TDMA MPI 합 (모멘텀·스칼라·압력 z) + halo 동기화, 구현별 (본 측정 런의 상위 구간만)."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.2 * len(sizes), 5.0), dpi=150)
    for ax, size, t in zip(axes, sizes, "abcd"):
        allnp = set()
        for dec in ("pencil", "slab"):
            dd = d[(size, dec)]
            nps = sorted(dd)
            if not nps:
                continue
            tdma = [sum(v for k, v in dd[n]["med"].items() if k.startswith("tdma_comm_")) for n in nps]
            halo = [sum(dd[n]["med"].get(k, 0.0) for k in sp.HALO) for n in nps]
            ax.plot(nps, tdma, color=COL[dec], marker=MK[dec], ms=8, lw=2, label=LAB[dec] + " — TDMA MPI", zorder=3)
            ax.plot(nps, halo, color=COL[dec], marker=MK[dec], ms=8, lw=2, ls=":", mfc="white",
                    label=LAB[dec] + " — halo", zorder=3)
            allnp |= set(nps)
        if not allnp:
            ax.set_visible(False)
            continue
        axis_np(ax, sorted(allnp))
        ax.set_ylim(0, None)
        ax.set_ylabel("time per step [ms]")
        n = size.split("_")[1]
        tag(ax, f"({t}) {size.split('_')[0]} {n}³" + ("/GPU" if size.startswith("weak") else ""))
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


def pressure_internals_fig(dph, size, fname):
    """phases 런: 압력 내부 묶음 (전치 MPI / 팩 / FFT / z TDMA / 기타) 누적 막대, np 마다 pencil·slab 쌍."""
    p, s = dph.get((size, "pencil"), {}), dph.get((size, "slab"), {})
    nps = sorted(set(p) | set(s))
    if not nps:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(nps) + 5.5, 5.0), dpi=150)
    shade = {"pencil": 1.0, "slab": 0.55}
    names = None
    w = 0.38
    for si, (dec, dd) in enumerate((("pencil", p), ("slab", s))):
        for xi, n in enumerate(nps):
            if n not in dd:
                continue
            groups = pressure_groups(dd[n]["med"])
            names = [g for g, _ in groups]
            bottom = 0.0
            x = xi + (si - 0.5) * w
            for gi, (_, v) in enumerate(groups):
                ax.bar(x, v, width=w * 0.92, bottom=bottom, color=sp.SLOTS[gi] if gi < 4 else sp.NEUTRAL,
                       alpha=shade[dec], edgecolor=sp.SURFACE, linewidth=1.0, zorder=3)
                bottom += v
            ax.annotate(dec[0].upper(), (x, bottom), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=10, color=sp.INK2)
    ax.set_xticks(range(len(nps)))
    ax.set_xticklabels([str(n) for n in nps])
    ax.set_xlabel("GPUs   (P = pencil, S = slab)")
    ax.set_ylabel("pressure solve [ms]")
    ax.grid(True, axis="y", color=sp.GRID, lw=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=sp.SLOTS[i] if i < 4 else sp.NEUTRAL) for i in range(len(names))], names, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    n = size.split("_")[1]
    tag(ax, f"{size.split('_')[0]} {n}³/GPU")
    fig.tight_layout()
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


def table(d, path):
    lines = ["# 논문 세트 — 지금까지의 결과 (ms/step, 랭크 max, 3 회 중앙값 런; [min–max]) — results/paper_zyx", ""]
    for size in ("weak_256", "weak_512", "strong_1024", "strong_2048"):
        p, s = d[(size, "pencil")], d[(size, "slab")]
        nps = sorted(set(p) | set(s))
        if not nps:
            continue
        b = BASE[size]
        lines += [f"## {size}", "", "| np | pencil | slab | slab/pencil | E pencil | E slab | pressure share P / S |",
                  "|---|---|---|---|---|---|---|"]
        for n in nps:
            def cell(dd):
                if n not in dd:
                    return "—", None
                r = dd[n]
                return "%.1f [%.1f–%.1f]" % (r["med"]["step_total"], min(r["all"]), max(r["all"])), r
            cp, rp = cell(p)
            cs, rs = cell(s)
            def eff(dd, r):
                if r is None or b not in dd:
                    return "—"
                t0 = dd[b]["med"]["step_total"]
                return "%.2f" % (t0 / r["med"]["step_total"] if size.startswith("weak")
                                 else t0 * b / (r["med"]["step_total"] * n))
            ratio = "%.2f" % (rs["med"]["step_total"] / rp["med"]["step_total"]) if rp and rs else "—"
            sh = "%.2f / %.2f" % (rp["med"].get("pressure_solve", 0) / rp["med"]["step_total"] if rp else 0,
                                  rs["med"].get("pressure_solve", 0) / rs["med"]["step_total"] if rs else 0)
            lines.append(f"| {n} | {cp} | {cs} | {ratio} | {eff(p, rp)} | {eff(s, rs)} | {sh} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    os.chdir(APP)
    FIG.mkdir(parents=True, exist_ok=True)
    d = load(RES)
    dph = load(RES_PH)
    weak = [s for s in ("weak_256", "weak_512") if d[(s, "pencil")] or d[(s, "slab")]]
    strong = [s for s in ("strong_1024", "strong_2048") if d[(s, "pencil")] or d[(s, "slab")]]
    if weak:
        time_fig(d, weak, "F1_weak_time")
        eff_fig(d, weak, "F2_weak_efficiency")
        share_fig(d, weak, "F6_weak_pressure_share")
        comm_fig(d, weak, "F8_weak_comm")
    if strong:
        time_fig(d, strong, "F4_strong_time")
        eff_fig(d, strong, "F5_strong_efficiency")
        share_fig(d, strong, "F7_strong_pressure_share")
    ratio_fig(d, "F3_slab_over_pencil")
    for size in ("weak_256", "weak_512"):
        pressure_internals_fig(dph, size, f"F9_{size}_pressure_internals")
    table(d, FIG / "paper_table.md")
    print((FIG / "paper_table.md").read_text())


if __name__ == "__main__":
    main()
