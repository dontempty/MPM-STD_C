# 압력 솔버 그림 P1–P6 — plot/figures/pressure/ (weak 512³/GPU, cube 블록, 평면 경로, 4 ranks/node)
#   최신 결과로 np 1–256 을 한 줄로 잇는다:
#     np 1–4   results/weak512_zyx_plane(_phases)      (잡 1201468–1201487, 2026-11-02)
#     np 8–256 results/weak512_zyx_plane256(_phases)   (잡 1204156–1204205, 2026-11-03)  ← np 8 은 이쪽
#   참고선: results/weak512_zyx_selfcopy (4 회 전치, np 8–64, 평면 경로 이전)
#   P1 pressure_solve 시간, P2 압력 효율 (기준 np 8), P3 스텝 중 압력 비중, P4 압력 내부 분해 (누적 막대),
#   P5 압력 내부 항목별 선 (전치 MPI / z TDMA MPI / 계산), P6 np 256 손실 순위 (압력 내부 단계)
#   + pressure_table.md (숫자 표)
# 사용 (apps/rbc_nob_scalability/ 에서):  /scratch/t525a06/KDH/.venv-plot/bin/python plot/present_pressure.py
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
from present_np1_64 import pressure_groups, pressure_bars

st = "weak_512_pencil"
PLANE, PLANE256, BEFORE = "results/weak512_zyx_plane", "results/weak512_zyx_plane256", "results/weak512_zyx_selfcopy"
PH_PLANE, PH_PLANE256 = "results/weak512_zyx_plane_phases", "results/weak512_zyx_plane256_phases"
DECOMP = {1: "1x1x1", 2: "1x1x2", 4: "1x2x2", 8: "2x2x2", 16: "2x2x4", 32: "2x4x4", 64: "4x4x4", 128: "4x4x8",
          256: "4x8x8"}


def merged(lo_res, hi_res, split=8):
    """np < split 은 lo_res, np >= split 은 hi_res (같은 np 가 둘 다 있으면 최신인 hi_res)."""
    d = {n: r for n, r in sp.collect([lo_res], st).items() if n < split}
    d.update({n: r for n, r in sp.collect([hi_res], st).items() if n >= split})
    return dict(sorted(d.items()))


def with_pressure(d, base=8):
    for r in d.values():
        r["pressure"] = r["phases"].get("pressure_solve", 0.0)
        r["share"] = r["pressure"] / r["total"]
    p0 = d[base]["pressure"] if base in d else d[min(d)]["pressure"]
    for r in d.values():
        r["peff"] = p0 / r["pressure"]
    return sp.metrics(d, "weak")


def internals(php):
    """{np: {"transpose MPI", "z TDMA MPI", "compute", "wait (mean, halo)"}} — 랭크 max 단계값."""
    out = {}
    for n, r in php.items():
        ph = r["phases"]
        g = dict(pressure_groups(ph))
        zmpi = ph.get("tdma_comm_pressure_z", 0.0)
        out[n] = {"transpose MPI": g["transpose MPI"], "z TDMA MPI": zmpi,
                  "compute": g["transpose pack/unpack"] + g["FFT + reorder"] + (g["z TDMA (build + solve)"] - zmpi),
                  "wait (mean, halo)": g["other (mean, halo)"], "place": r["place"]}
    return out


def internals_chart(php, fname):
    it = internals(php)
    keys = ("transpose MPI", "z TDMA MPI", "compute", "wait (mean, halo)")
    sets = {}
    for k in keys[:3]:
        sets[k] = {n: {"v": it[n][k], "place": it[n]["place"]} for n in it}
    sp.line_chart(sets, "v", title="Weak np %d–%d — inside pressure_solve, plane path (rank max)" % (min(it), max(it)),
                  ylabel="ms/step", logy=False, ylim=(0, max(v["v"] for d in sets.values() for v in d.values()) * 1.2),
                  labels={k: k for k in keys}, fmt="%.1f ms", legend_loc="upper left", fname=fname)


def loss_chart(php, fname):
    """np 최대 − np 8 의 압력 내부 단계별 증가량 (랭크 max)."""
    nps = sorted(php)
    n0, n1 = nps[0], nps[-1]
    names = [k for k in php[n1]["phases"] if k.startswith("pressure.") and not k.startswith("pressure.comm.")]
    rows = sorted(((k, php[n1]["phases"][k] - php[n0]["phases"].get(k, 0.0), php[n0]["phases"].get(k, 0.0),
                    php[n1]["phases"][k]) for k in names), key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(rows) + 1.4), dpi=130)
    ys = list(range(len(rows)))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(ys, vals, height=0.62, color=sp.SLOTS[0], edgecolor=sp.SURFACE, linewidth=1.1, zorder=3)
    span = max(abs(v) for v in vals) or 1.0
    for y, (k, loss, t0, t1) in zip(ys, rows):
        ax.annotate("%+.1f ms   (%.1f → %.1f)" % (loss, t0, t1), xy=(max(loss, 0), y), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5, color=sp.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0].replace("pressure.", "") for r in rows], fontsize=8.5)
    ax.set_xlim(min(0, min(vals)) - 0.02 * span, span * 1.6)
    ax.axvline(0, color=sp.INK2, lw=0.8, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("increase np %d → %d  [ms/step, rank max]" % (n0, n1))
    ax.set_title("Weak — pressure_solve stages, growth from np %d to %d (plane path)" % (n0, n1), loc="left")
    total = php[n1]["phases"]["pressure_solve"] - php[n0]["phases"]["pressure_solve"]
    ax.annotate("pressure_solve %.1f → %.1f ms  (+%.1f)" % (php[n0]["phases"]["pressure_solve"],
                                                             php[n1]["phases"]["pressure_solve"], total),
                xy=(1, 0), xycoords="axes fraction", xytext=(0, -34), textcoords="offset points", ha="right",
                va="top", fontsize=9, color=sp.INK2)
    fig.tight_layout()
    sp._save(fig, fname)


def write_table(d, php, b, path):
    lines = ["# 압력 솔버 weak 512³/GPU — 최신 결과 (plane path; np 1–4 = 2026-11-02 잡 12014xx, np 8–256 = 2026-11-03 잡 12041xx)",
             "", "ms/step, 랭크 max. 효율 E_p = pressure(8) / pressure(np). 참고 = 4 회 전치 경로 (2026-11-02).", "",
             "| np | 분할 | 노드×랭크 | step | pressure | 비중 | E_p | 참고 pressure (4 전치) | 전치 MPI | z TDMA MPI | 계산 | 대기 | 잡 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    it = internals(php)
    for n, r in d.items():
        i = it.get(n)
        bb = b.get(n)
        lines.append("| %d | %s | %s | %.1f | %.1f | %.0f %% | %.2f | %s | %s | %s | %s | %s | %s |" % (
            n, DECOMP.get(n, "?"), r["place"], r["total"], r["pressure"], 100 * r["share"], r["peff"],
            "%.1f" % bb["pressure"] if bb else "—",
            "%.1f" % i["transpose MPI"] if i else "—", "%.1f" % i["z TDMA MPI"] if i else "—",
            "%.1f" % i["compute"] if i else "—", "%.1f" % i["wait (mean, halo)"] if i else "—",
            r["file"].split("_j")[-1].replace(".csv", "")))
    lines += ["", "내부 항목은 solver_phases 런(별도 잡, 1 회)의 단계 타이머라 step·pressure 열(타이머 없는 런)과 잡이 다르다.",
              "전치 MPI = pressure.comm.block_to_planes + planes_to_block, z TDMA MPI = tdma_comm_pressure_z,",
              "계산 = 전치 팩·언팩 + FFT·재배열 + build_z + z TDMA 국소, 대기 = mean_remove + halo.",
              "", "그림: P1 시간, P2 효율, P3 스텝 비중, P4 내부 누적 막대, P5 내부 항목별 선, P6 np 8→256 단계별 증가.",
              "스크립트: plot/present_pressure.py. 이전 그림(A·B, 2026-11-02, np ≤ 64)은 old_20261102/."]
    path.write_text("\n".join(lines) + "\n")


def main():
    os.chdir(APP)
    sp.FIGDIR = APP / "plot" / "figures" / "pressure"
    d = with_pressure(merged(PLANE, PLANE256))
    b = with_pressure({n: r for n, r in sp.collect([BEFORE], st).items() if 8 <= n <= 64})
    php = {}
    php.update({n: r for n, r in sp.collect([PH_PLANE], st).items() if n < 8})
    php.update({n: r for n, r in sp.collect([PH_PLANE256], st).items() if n >= 8})
    php = dict(sorted(php.items()))

    sets = {st + "@p": d, st + "@b": b}
    lab = {st + "@p": "plane path (2 transposes), latest", st + "@b": "before: 4 transposes (np 8–64)"}
    print("== pressure, np 1-256"); sp.table(sets)
    ptop = max(r["pressure"] for s in sets.values() for r in s.values())
    sp.line_chart(sets, "pressure", title="Weak np 1–256 — pressure_solve per step (512³/GPU, cube blocks, 4 ranks/node)",
                  ylabel="pressure_solve [ms/step]  (rank max)", logy=False, ylim=(0, ptop * 1.2), labels=lab,
                  fmt="%.1f ms", legend_loc="upper left", fname="P1_np1-256_pressure_time")
    d8 = {n: r for n, r in d.items() if n >= 8}
    sp.line_chart({st + "@p": d8, st + "@b": b}, "peff",
                  title="Weak np 8–256 — pressure_solve efficiency  E_p = P(8) / P(np)",
                  ylabel="pressure efficiency (base np 8)", logy=False, labels=lab, ideal=lambda x, dd: 1.0,
                  ideal_label="ideal  1.0", ylim=(0, 1.15), fmt="%.2f", legend_loc="lower left",
                  fname="P2_np8-256_pressure_efficiency")
    sp.line_chart({st + "@p": d}, "share", title="Weak np 1–256 — pressure_solve share of the step",
                  ylabel="pressure_solve / step_total", logy=False, ylim=(0, 1.0), labels=lab, fmt="%.0f %%" if False
                  else "%.2f", legend_loc="upper left", fname="P3_np1-256_pressure_share")
    pressure_bars([("plane", php)], sorted(php), "Weak np 1–256 — inside pressure_solve, plane path",
                  "P4_np1-256_pressure_breakdown")
    internals_chart({n: r for n, r in php.items() if n >= 8}, "P5_np8-256_pressure_internals")
    loss_chart({n: r for n, r in php.items() if n >= 8}, "P6_np8-256_pressure_stage_growth")
    write_table(d, php, b, sp.FIGDIR / "pressure_table.md")

    print("\n== internals (phases runs)")
    for n, i in internals(php).items():
        print("%4d  transpose MPI %6.1f  z MPI %5.1f  compute %5.1f  wait %4.1f" % (
            n, i["transpose MPI"], i["z TDMA MPI"], i["compute"], i["wait (mean, halo)"]))


if __name__ == "__main__":
    main()
