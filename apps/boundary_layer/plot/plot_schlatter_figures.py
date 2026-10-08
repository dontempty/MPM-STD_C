#!/usr/bin/env python3
"""Schlatter & Örlü (2010, JFM 659) 논문 그림 형식으로 우리 bl_stats 를 참조와 함께 그린다.
사용: python3 plot_schlatter_figures.py stats/re4500/bl_stats_X.dat [stats/re6000/bl_stats_Y.dat ...] [--out plot/schlatter]
  <out>_integral.png : fig.2 스타일 — c_f(Re_θ), H(Re_θ), Re_τ(Re_θ): 우리 x-분포(선) + KTH·Torroja 점 + 상관식
  <out>_mean.png     : fig.3 스타일 — 스테이션별 U⁺(y⁺) vs 참조 + 로그 법칙, 아래 줄은 U⁺ − U⁺_ref
  <out>_rms.png      : fig.4 스타일 — 스테이션별 u,v,w rms⁺ 와 −uv⁺ vs 참조
참조: ref/vel_{670,1000,1410,2000}_dns.prof (KTH), ref/torroja/Re_theta.{1100,1551,1968}.prof (Jiménez et al. 2010)
상관식: Smits et al. 1983 c_f = 0.024 Re_θ^{-1/4}, Coles–Fernholz c_f = 2[ln(Re_θ)/0.384 + 4.127]^{-2},
       Schlatter & Örlü 2010 Re_τ ≈ 1.13 Re_θ^{0.843}. 로그 법칙 κ = 0.41, B = 5.2."""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_schlatter import load_refs, read_stats, cf_correlations  # noqa: E402

KAPPA, B_LOG = 0.41, 5.2


def nearest_ref(refs, re_theta):
    return min(refs, key=lambda r: abs(r[0]["re_theta"] - re_theta))


def label_of(path):
    m = re.search(r"stats/([^/]+)/", path)
    case = m.group(1) if m else os.path.basename(os.path.dirname(path))
    h = open(path).readline()
    t = float(re.search(r"\bt=([0-9.eE+-]+)", h).group(1))
    acc = float(re.search(r"accumulated_time=([0-9.eE+-]+)", h).group(1))
    tag = f"t={t:.0f}, avg {acc:.0f}" if acc >= 1 else f"t={t:.0f}, snapshot"
    return case, tag


def main():
    argv = sys.argv[1:]
    out = "schlatter"
    if "--out" in argv:
        k = argv.index("--out"); out = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    refs = load_refs()
    cases = []
    for p in argv:
        integ, stations = read_stats(p)
        case, tag = label_of(p)
        cases.append(dict(path=p, case=case, tag=tag, integ=integ, stations=stations))
    colors = {c["case"]: plt.cm.tab10(i) for i, c in enumerate(cases)}

    # ── fig.2 스타일: 적분량 vs Re_θ ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    rt = np.logspace(np.log10(400), np.log10(4500), 200)
    smits, cfz = cf_correlations(rt)
    ax[0].plot(rt, smits, "k--", lw=1, label="Smits et al. 1983: 0.024 Re_θ^-1/4")
    ax[0].plot(rt, cfz, "k:", lw=1.2, label="Coles–Fernholz (κ=0.384, C=4.127)")
    ax[2].plot(rt, 1.13 * rt ** 0.843, "k--", lw=1, label="S&Ö 2010 fit: 1.13 Re_θ^0.843")
    for src, mk in (("KTH", "s"), ("Torroja", "D")):
        pts = [r[0] for r in refs if r[0]["src"] == src]
        if not pts: continue
        x = [p["re_theta"] for p in pts]
        ax[0].plot(x, [p["cf"] for p in pts], mk, color="k", mfc="white", ms=7, label=f"{src} DNS")
        ax[1].plot(x, [p["H"] for p in pts], mk, color="k", mfc="white", ms=7, label=f"{src} DNS")
        ax[2].plot(x, [p["re_tau"] for p in pts], mk, color="k", mfc="white", ms=7, label=f"{src} DNS")
    for c in cases:
        g = c["integ"]; col = colors[c["case"]]
        keep = g["re_theta"] > 0
        ax[0].plot(g["re_theta"][keep], g["cf"][keep], "-", color=col, lw=1.2, alpha=0.9, label=f"MPM-STD {c['case']} ({c['tag']})")
        ax[1].plot(g["re_theta"][keep], g["H"][keep], "-", color=col, lw=1.2, alpha=0.9)
        ax[2].plot(g["re_theta"][keep], g["re_tau"][keep], "-", color=col, lw=1.2, alpha=0.9)
        for st in c["stations"]:
            m = st["meta"]
            ax[0].plot(m["Re_theta"], m["Cf"], "o", color=col, ms=6)
            ax[1].plot(m["Re_theta"], m["H"], "o", color=col, ms=6)
            ax[2].plot(m["Re_theta"], m["Re_tau"], "o", color=col, ms=6)
    ax[0].set_ylabel(r"$c_f$"); ax[0].set_ylim(0.002, 0.007)
    ax[1].set_ylabel(r"$H_{12}$"); ax[1].set_ylim(1.3, 1.7)
    ax[2].set_ylabel(r"$Re_\tau$"); ax[2].set_yscale("log")
    for a in ax:
        a.set_xscale("log"); a.set_xlim(400, 4500); a.set_xlabel(r"$Re_\theta$"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.suptitle("ZPG turbulent boundary layer — integral quantities vs Re_θ (Schlatter & Örlü 2010 fig. 2 style; "
                 "lines = our x-distribution, dots = stations)", fontsize=10)
    fig.tight_layout(); fig.savefig(out + "_integral.png", dpi=140); print("saved", out + "_integral.png")

    # ── fig.3/4 스타일: 스테이션 프로파일 ────────────────────────────────────────
    all_st = [(c, st) for c in cases for st in c["stations"]]
    if not all_st:
        return
    n = len(all_st)
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 8.2), squeeze=False)
    yp_line = np.logspace(0, 3.5, 200)
    for q, (c, st) in enumerate(all_st):
        m, pr = st["meta"], st["prof"]; col = colors[c["case"]]
        rm, rp = nearest_ref(refs, m["Re_theta"])
        a = ax[0, q]
        a.plot(rp["yp"], rp["U"], "k-", lw=1.6, label=f"{rm['src']} Re_θ={rm['re_theta']:.0f}")
        a.plot(pr["yp"], pr["U"], "-", color=col, lw=1.3, label=f"MPM-STD {c['case']} Re_θ={m['Re_theta']:.0f}")
        a.plot(yp_line, np.log(yp_line) / KAPPA + B_LOG, "k:", lw=0.9, label="log law (0.41, 5.2)")
        a.plot(yp_line, yp_line, "k-.", lw=0.7)
        a.set_xscale("log"); a.set_xlim(0.5, 2000); a.set_ylim(0, 27); a.set_xlabel(r"$y^+$"); a.set_ylabel(r"$U^+$")
        a.set_title(f"{c['case']}: Re_θ {m['Re_theta']:.0f} (Re_τ {m['Re_tau']:.0f}, c_f {m['Cf']:.4f}, H {m['H']:.3f})\n"
                    f"ref {rm['src']} Re_θ {rm['re_theta']:.0f} (Re_τ {rm['re_tau']:.0f}, c_f {rm['cf']:.4f}, H {rm['H']:.3f})", fontsize=8)
        a.legend(fontsize=7); a.grid(alpha=0.3)
        b = ax[1, q]
        U_ref = np.interp(pr["yp"], rp["yp"], rp["U"])
        b.plot(pr["yp"], pr["U"] - U_ref, "-", color=col, lw=1.3)
        b.axhspan(-0.3, 0.3, color="green", alpha=0.12); b.axhline(0, color="k", lw=0.8)
        b.set_xscale("log"); b.set_xlim(0.5, 2000); b.set_ylim(-2, 2); b.set_xlabel(r"$y^+$"); b.set_ylabel(r"$U^+ - U^+_{ref}$")
        b.grid(alpha=0.3)
    fig.suptitle("Mean velocity in inner scaling at the comparison stations (Schlatter & Örlü 2010 fig. 3 style)", fontsize=10)
    fig.tight_layout(); fig.savefig(out + "_mean.png", dpi=140); print("saved", out + "_mean.png")

    fig, ax = plt.subplots(n, 4, figsize=(17, 3.8 * n), squeeze=False)
    for q, (c, st) in enumerate(all_st):
        m, pr = st["meta"], st["prof"]; col = colors[c["case"]]
        rm, rp = nearest_ref(refs, m["Re_theta"])
        for p, (key, lab) in enumerate((("urms", r"$u^+_{rms}$"), ("vrms", r"$v^+_{rms}$ (span)"),
                                        ("wrms", r"$w^+_{rms}$ (wall-normal)"), ("uv", r"$-\overline{uw}^+$"))):
            a = ax[q, p]
            # KTH/Torroja: v = 벽수직, w = 스팬 (우리는 v = 스팬, w = 벽수직) → 교환
            ref_key = {"urms": "urms", "vrms": "wrms", "wrms": "vrms", "uv": "uv"}[key]
            ref_val = rp[ref_key] if key != "uv" else -rp["uv"]
            our_val = pr[key] if key != "uv" else -pr["uv"]
            a.plot(rp["yp"], ref_val, "k-", lw=1.6, label=f"{rm['src']} Re_θ={rm['re_theta']:.0f}")
            a.plot(pr["yp"], our_val, "-", color=col, lw=1.3, label=f"MPM-STD {c['case']} Re_θ={m['Re_theta']:.0f}")
            a.set_xscale("log"); a.set_xlim(0.5, 2000); a.set_xlabel(r"$y^+$"); a.set_ylabel(lab); a.grid(alpha=0.3)
            if p == 0: a.legend(fontsize=7)
    fig.suptitle("Reynolds stresses in inner scaling at the comparison stations (Schlatter & Örlü 2010 fig. 4 style)", fontsize=10)
    fig.tight_layout(); fig.savefig(out + "_rms.png", dpi=140); print("saved", out + "_rms.png")


if __name__ == "__main__":
    main()
