#!/usr/bin/env python3
"""통계 검증(원본 vs 현재) 시각화 노트북 — apps/<app>/plot_validate.ipynb (dhvc: Ra 1e7, rbc_nob: ΔT 40). 그림은 docs/Scalability/validation/ 에 PNG 로 저장(증거).
   사용: /usr/bin/python3 tools/scale/build_validate_plot.py dhvc"""
import base64, contextlib, io, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def dhvc_cells():
    md = lambda s: ('markdown', s); code = lambda s: ('code', s)
    return [
        md("""# DHVC Ra = 1e7 — 최적화 전(원본 f330720) vs 후(현재) 통계 비교

`input_dhvc_1e7_val_{orig,new}.ini`: 832×416×192, npz 4 (V100 ×4, gpu19 / gpu18), 1e6 검증 런(t 240) 체크포인트를 같은 warm-start 로 써서 t 200 까지, 통계 창 **t 100~200** (누적 파일 125·150·175·200).
원본 = `build_gpu/bin/dhvc_validate_orig`(원본 코드 + 계측 main, 타이머 끔), 현재 = `dhvc_validate_new`. 두 런은 같은 초기장이지만 반올림 차이로 곧 다른 난류 실현으로 갈라지므로 **창 변동 안의 차이면 같은 통계**로 본다.
그림은 증거로 `docs/Scalability/validation/dhvc_1e7_*.png` 에 저장한다."""),
        code(r'''import glob, re, os
import numpy as np
import matplotlib
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
SAVE_FIG = True
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "docs", "Scalability", "validation"); os.makedirs(FIG_DIR, exist_ok=True)
def save(fig, name):
    if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, name), dpi=130, bbox_inches="tight"); print("saved", os.path.relpath(os.path.join(FIG_DIR, name), os.getcwd()))
COLS = ["z", "U", "V", "W", "T", "urms", "vrms", "wrms", "trms", "uw", "uT", "wT"]
STYLE = {"orig": dict(color="#9a9a96", ls="--", marker="o", mfc="white", ms=5, mew=1.2), "new": dict(color="#2a78d6", ls="-", marker="s", ms=3)}
def load(path):
    hdr, rows = {}, []
    for l in open(path):
        if l.startswith("#"):
            for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", l): hdr[k] = float(v)
        elif l.strip(): rows.append([float(x) for x in l.split()])
    return hdr, np.array(rows)
RUNS = {v: sorted(glob.glob(f"out_val_1e7_{v}/dhvc_stats_*.dat")) for v in ("orig", "new")}
FINAL = {v: load(f[-1]) for v, f in RUNS.items()}
SERIES = {v: [load(f)[0] for f in fs] for v, fs in RUNS.items()}
TITLE = "DHVC Ra 1e7, 832×416×192, V100 ×4 — window t 100–200"
print(f"{'quantity':16s} {'orig':>12s} {'new':>12s} {'diff':>8s}")
for k in ("Nu_hot", "Nu_cold", "u_tau_hot", "u_tau_cold", "Re_dstar", "Re_rms", "U_rms_delta_max", "delta_star", "delta_max"):
    a, b = FINAL["orig"][0][k], FINAL["new"][0][k]; print(f"{k:16s} {a:12.5g} {b:12.5g} {100*(b-a)/a:7.2f}%")'''),
        md("## 1. 평균 프로파일 ⟨U⟩(z), ⟨T⟩(z) — 원본(○ 파선) vs 현재(■ 실선)"),
        code(r'''fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for v in ("orig", "new"):
    h, d = FINAL[v]; z = d[:, 0]
    axes[0].plot(d[:, 1], z, label=f"{v}  Re_δ* {h['Re_dstar']:.2f}", **STYLE[v])
    axes[1].plot(d[:, 4], z, label=f"{v}  Nu {h['Nu_hot']:.2f}/{h['Nu_cold']:.2f}", **STYLE[v])
    axes[2].semilogx(z, d[:, 4], label=v, **STYLE[v])
axes[0].set_xlabel("⟨U⟩"); axes[0].set_ylabel("z"); axes[1].set_xlabel("⟨T⟩"); axes[2].set_xlabel("z (log)"); axes[2].set_ylabel("⟨T⟩  (wall layer)")
for ax in axes: ax.legend(frameon=False)
fig.suptitle(f"{TITLE}: mean profiles", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "dhvc_1e7_mean_profiles.png"); plt.show()'''),
        md("## 2. rms 프로파일 u·v·w·T"),
        code(r'''fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for i, (ax, c) in enumerate(zip(axes, ("urms", "vrms", "wrms", "trms"))):
    for v in ("orig", "new"):
        d = FINAL[v][1]; ax.plot(d[:, 5 + i], d[:, 0], label=v, **STYLE[v])
    ax.set_xlabel(c); ax.set_ylabel("z" if i == 0 else ""); ax.legend(frameon=False)
fig.suptitle(f"{TITLE}: rms profiles", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "dhvc_1e7_rms_profiles.png"); plt.show()'''),
        md("## 3. 플럭스 ⟨u'w'⟩, ⟨u'T'⟩, ⟨w'T'⟩"),
        code(r'''fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for i, (ax, c) in enumerate(zip(axes, ("uw", "uT", "wT"))):
    for v in ("orig", "new"):
        d = FINAL[v][1]; ax.plot(d[:, 9 + i], d[:, 0], label=v, **STYLE[v])
    ax.set_xlabel(c); ax.set_ylabel("z" if i == 0 else ""); ax.legend(frameon=False)
fig.suptitle(f"{TITLE}: turbulent fluxes", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "dhvc_1e7_fluxes.png"); plt.show()'''),
        md("## 4. 누적 창(100 → t)에 따른 Nu·Re 수렴 — 창 변동 폭이 전/후 차이의 척도"),
        code(r'''fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for v in ("orig", "new"):
    t = [h["t_start"] + h["sum_dt"] for h in SERIES[v]]
    axes[0].plot(t, [h["Nu_hot"] for h in SERIES[v]], label=f"{v} Nu_hot", **STYLE[v])
    axes[0].plot(t, [h["Nu_cold"] for h in SERIES[v]], label=f"{v} Nu_cold", alpha=0.6, **STYLE[v])
    axes[1].plot(t, [h["Re_rms"] for h in SERIES[v]], label=v, **STYLE[v])
    axes[2].plot(t, [h["Re_dstar"] for h in SERIES[v]], label=v, **STYLE[v])
axes[0].set_ylabel("Nu"); axes[1].set_ylabel("Re_rms"); axes[2].set_ylabel("Re_δ*")
for ax in axes: ax.set_xlabel("window end t (start 100)"); ax.legend(frameon=False)
fig.suptitle(f"{TITLE}: cumulative-window convergence", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "dhvc_1e7_convergence.png"); plt.show()
print("profile max|diff|/max|orig|:", ", ".join(f"{c} {100*np.max(np.abs(FINAL['new'][1][:, i]-FINAL['orig'][1][:, i]))/max(np.max(np.abs(FINAL['orig'][1][:, i])), 1e-300):.2f}%" for i, c in enumerate(COLS) if c in ("U", "T", "urms", "vrms", "wrms", "trms", "uw", "wT")))'''),
    ]

def rbc_cells():
    md = lambda s: ('markdown', s); code = lambda s: ('code', s)
    return [
        md("""# RBC NOB ΔT = 40 K (cv12) — 최적화 전(원본 f330720) vs 후(현재) 통계 비교

`input_rbc_nob_cv12_dt40_val_{orig,new}.ini`: 128×128×256, Ra 1e6, Pr 2547, npz 4 (V100 ×4, gpu17 / gpu16), 같은 냉시작(결정론적 섭동)에서 t 25000 까지, 통계 창 **t 16900~25000** (누적 파일 1000 마다).
원본 = `build_gpu/bin/rbc_nob_validate_orig`(원본 코드 + 계측 main), 현재 = `rbc_nob_validate_new`. 참조 = 이전 클러스터 완주(잡 36135, npz 2): θ_c 0.13852 / Nu 9.597, Horn 2013 T_c +0.2 %.
그림은 `docs/Scalability/validation/rbc_dt40_*.png` 에 저장."""),
        code(r"""import glob, re, os
import numpy as np
import matplotlib
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
SAVE_FIG = True
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "docs", "Scalability", "validation"); os.makedirs(FIG_DIR, exist_ok=True)
def save(fig, name):
    if SAVE_FIG: fig.savefig(os.path.join(FIG_DIR, name), dpi=130, bbox_inches="tight"); print("saved", os.path.relpath(os.path.join(FIG_DIR, name), os.getcwd()))
STYLE = {"orig": dict(color="#9a9a96", ls="--", marker="o", mfc="white", ms=5, mew=1.2), "new": dict(color="#2a78d6", ls="-", marker="s", ms=3)}
REF = {"theta_c": 0.13852, "Nu": 9.597}
LOGS = {"orig": sorted(glob.glob("log/dt40_orig_*.log")), "new": sorted(glob.glob("log/dt40_new_*.log"))}
def load_stats(path):
    hdr, rows = {}, []
    for l in open(path):
        if l.startswith("#"):
            for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", l): hdr[k] = float(v)
        elif l.strip(): rows.append([float(x) for x in l.split()])
    return hdr, np.array(rows)
def monitor(log):
    t, th, nub, nut = [], [], [], []
    for l in open(log):
        if re.match(r"^\s*\d+\s", l):
            f = l.split(); t.append(float(f[1])); th.append(float(f[4])); nub.append(float(f[8])); nut.append(float(f[9]))
    return np.array(t), np.array(th), np.array(nub), np.array(nut)
RUNS = {v: sorted(glob.glob(f"stats_val_dt40_{v}/rbc_nob_stats_*.dat")) for v in ("orig", "new")}
HAVE = [v for v in ("orig", "new") if RUNS[v]]
FINAL = {v: load_stats(RUNS[v][-1]) for v in HAVE}
SERIES = {v: [load_stats(f)[0] for f in RUNS[v]] for v in HAVE}
MON = {v: monitor(LOGS[v][-1]) for v in ("orig", "new") if LOGS[v]}
TITLE = "RBC NOB ΔT 40, 128×128×256, V100 ×4"
keys = ["theta_c", "Nu_volume", "Nu_bottom", "Nu_top", "lambda_theta_b", "lambda_theta_t", "chi_lambda_theta", "Re_E_m", "Re_E_c", "Re_pl_m", "T_rms_center"]
print(f"{'quantity':18s}" + "".join(f"{v:>12s}" for v in HAVE) + ("   diff" if len(HAVE) == 2 else "") + "   (window t_start→t)")
for k in keys:
    row = f"{k:18s}" + "".join(f"{FINAL[v][0][k]:12.5g}" for v in HAVE)
    if len(HAVE) == 2: row += f"  {100*(FINAL['new'][0][k]-FINAL['orig'][0][k])/FINAL['orig'][0][k]:+6.2f}%"
    print(row)
for v in HAVE: print(f"   {v}: window {FINAL[v][0]['t_start']:.0f}→{FINAL[v][0]['t']:.0f} (sum_dt {FINAL[v][0]['sum_dt']:.0f})")
print(f"reference (old cluster, npz 2, window 16900→25000): theta_c {REF['theta_c']} / Nu {REF['Nu']}")"""),
        md("## 1. 시간 이력 — 모니터의 θ_c, Nu_b/Nu_t (전 구간; 음영 = 통계 창)"),
        code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for v, (t, th, nub, nut) in MON.items():
    axes[0].plot(t, th, label=v, **{**STYLE[v], "marker": None}); axes[1].plot(t, 0.5 * (nub + nut), label=f"{v} (Nu_b+Nu_t)/2", **{**STYLE[v], "marker": None})
axes[0].axhline(REF["theta_c"], color="#e34948", lw=0.8, ls=":", label="reference 0.13852"); axes[1].axhline(REF["Nu"], color="#e34948", lw=0.8, ls=":", label="reference 9.597")
for ax, yl in zip(axes, ("theta_c (instantaneous)", "Nu wall (instantaneous)")):
    ax.axvspan(16900, 25000, color="#2a78d6", alpha=0.06); ax.set_xlabel("t"); ax.set_ylabel(yl); ax.legend(frameon=False)
axes[0].set_ylim(0.10, 0.16); axes[1].set_ylim(8, 11)
fig.suptitle(f"{TITLE}: histories (same cold start, two binaries)", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "rbc_dt40_history.png"); plt.show()
# 결정론적 구간: 두 로그가 출력 정밀도에서 같은 마지막 t
if len(MON) == 2:
    m = min(len(MON["orig"][0]), len(MON["new"][0])); d = np.abs(MON["new"][1][:m] - MON["orig"][1][:m])
    k = np.argmax(d > 0) if (d > 0).any() else None
    print("theta_c identical at print precision until t =", f"{MON['orig'][0][k-1]:.0f}" if k else "end of shorter log")"""),
        md("## 2. 통계 창 프로파일 — ⟨T⟩(z), u/v/w/T rms, ⟨wT⟩, Nu_z: 원본(○ 파선) vs 현재(■ 실선)"),
        code(r"""cols = ["z", "T_mean", "u_rms", "v_rms", "w_rms", "T_rms", "wT_mean", "Nu_z", "kappa_mean", "mu_mean", "ke_mean"]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, c in zip(axes.flat, ["T_mean", "u_rms", "v_rms", "w_rms", "T_rms", "wT_mean", "Nu_z", "ke_mean"]):
    i = cols.index(c)
    for v in HAVE:
        d = FINAL[v][1]; ax.plot(d[:, i], d[:, 0], label=v, **STYLE[v])
    ax.set_xlabel(c); ax.set_ylabel("z"); ax.legend(frameon=False)
fig.suptitle(f"{TITLE}: window-averaged profiles (t {FINAL[HAVE[0]][0]['t_start']:.0f}–{FINAL[HAVE[0]][0]['t']:.0f})", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "rbc_dt40_profiles.png"); plt.show()
if len(HAVE) == 2:
    print("profile max|diff|/max|orig|:", ", ".join(f"{c} {100*np.max(np.abs(FINAL['new'][1][:, cols.index(c)]-FINAL['orig'][1][:, cols.index(c)]))/max(np.max(np.abs(FINAL['orig'][1][:, cols.index(c)])), 1e-300):.2f}%" for c in ("T_mean", "u_rms", "w_rms", "T_rms", "wT_mean", "Nu_z")))"""),
        md("## 3. 누적 창(16900 → t)에 따른 θ_c·Nu·λ_θ·Re 수렴 — 창 변동 폭이 전/후 차이의 척도"),
        code(r"""fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for v in HAVE:
    t = [h["t"] for h in SERIES[v]]
    axes[0].plot(t, [h["theta_c"] for h in SERIES[v]], label=v, **STYLE[v])
    axes[1].plot(t, [h["Nu_volume"] for h in SERIES[v]], label=f"{v} Nu_volume", **STYLE[v]); axes[1].plot(t, [0.5*(h["Nu_bottom"]+h["Nu_top"]) for h in SERIES[v]], label=f"{v} Nu_wall", alpha=0.5, **STYLE[v])
    axes[2].plot(t, [h["chi_lambda_theta"] for h in SERIES[v]], label=v, **STYLE[v])
    axes[3].plot(t, [h["Re_E_m"] for h in SERIES[v]], label=v, **STYLE[v])
axes[0].axhline(REF["theta_c"], color="#e34948", lw=0.8, ls=":", label="reference"); axes[1].axhline(REF["Nu"], color="#e34948", lw=0.8, ls=":", label="reference")
for ax, yl in zip(axes, ("theta_c", "Nu", "chi_lambda_theta", "Re_E_m")): ax.set_xlabel("window end t (start 16900)"); ax.set_ylabel(yl); ax.legend(frameon=False)
fig.suptitle(f"{TITLE}: cumulative-window convergence", x=0.01, ha="left", fontsize=10); fig.tight_layout(); save(fig, "rbc_dt40_convergence.png"); plt.show()"""),
    ]

def run(app):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.chdir(os.path.join(ROOT, "apps", app)); ns = {"__name__": "__main__"}; out = []
    for kind, src in (rbc_cells() if app == 'rbc_nob' else dhvc_cells()):
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell["execution_count"] = None; outputs = []; buf = io.StringIO()
            with contextlib.redirect_stdout(buf): exec(compile(src, "<cell>", "exec"), ns)
            if buf.getvalue(): outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight")
                outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode()}})
            plt.close("all"); cell["outputs"] = outputs
        out.append(cell)
    nb = {"cells": out, "metadata": {"kernelspec": {"display_name": "base", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(ROOT, "apps", app, "plot_validate.ipynb"); json.dump(nb, open(path, "w"), ensure_ascii=False, indent=1)
    print("wrote", path, "figures", sum(1 for c in out for o in c.get("outputs", []) if o["output_type"] == "display_data"))

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "dhvc")
