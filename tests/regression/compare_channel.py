#!/usr/bin/env python3
"""
compare_channel.py — Compare MPM-STD_C and Filtered_TDMA channel statistics.

Both codes write Tecplot ASCII stats files with the header:
  TITLE = "..."
  VARIABLES = "Z" "Z_plus" "U_mean" "W_mean" "u_rms" "v_rms" "w_rms" "uw_stress" "P_mean"
  ZONE T="Stats", I=Nz, ...
  <data rows>

Usage
─────
  python3 tests/regression/compare_channel.py \
      --mpm   statistics_Re180/stats_final_00042000.dat \
      --ft    /path/to/Filtered_TDMA/channel/statistics/p1/stats_final_00030000.dat \
      --nu    3.5035e-4 \
      --out   tests/regression/channel_Re180_compare.png
"""

import argparse
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found — install with: pip3 install matplotlib", file=sys.stderr)
    sys.exit(1)


def load_stats(path: str):
    """Load Tecplot-ASCII stats file, skipping 3-line header."""
    data = np.loadtxt(path, skiprows=3)
    # columns: Z Z+ U_mean W_mean u_rms v_rms w_rms uw_stress P_mean
    return {
        "z":     data[:, 0],
        "zp":    data[:, 1],
        "Umean": data[:, 2],
        "Wmean": data[:, 3],
        "urms":  data[:, 4],
        "vrms":  data[:, 5],
        "wrms":  data[:, 6],
        "uw":    data[:, 7],
        "Pmean": data[:, 8],
    }


def utau_from_stats(s, nu: float) -> float:
    """Back-calculate u_tau from first point: u_tau = z+[0] * nu / z[0]."""
    return s["zp"][0] * nu / s["z"][0]


# Kim, Moser & Mansour (1987) DNS Re_tau=180 reference (selected points):
# Source: http://turbulence.ices.utexas.edu/MKM_1999.html (channel Re180 data)
# Columns: z+, U+, u_rms+, v_rms+, w_rms+, -uw+
_KMM_ZPLUS = np.array([
    0.090, 0.18, 0.35, 0.71, 1.42, 2.84, 5.68, 8.54, 11.4, 17.1,
    22.8, 28.5, 34.2, 45.4, 56.7, 68.0, 79.4, 90.7, 102., 113.,
    124., 136., 147., 158., 170., 181.
])
_KMM_UPLUS = np.array([
    0.081, 0.161, 0.323, 0.645, 1.29, 2.54, 4.64, 6.12, 7.16, 8.65,
    9.78, 10.7, 11.5, 12.9, 14.0, 14.9, 15.7, 16.4, 16.9, 17.4,
    17.9, 18.3, 18.6, 18.9, 19.2, 19.5
])


def main():
    ap = argparse.ArgumentParser(description="Compare MPM-STD_C vs Filtered_TDMA channel stats")
    ap.add_argument("--mpm",  required=True, help="MPM-STD_C stats file")
    ap.add_argument("--ft",   required=True, help="Filtered_TDMA stats file")
    ap.add_argument("--nu",   type=float, default=3.5035035e-4, help="kinematic viscosity")
    ap.add_argument("--out",  default="channel_Re180_compare.png", help="output PNG")
    args = ap.parse_args()

    mpm = load_stats(args.mpm)
    ft  = load_stats(args.ft)
    nu  = args.nu

    utau_mpm = utau_from_stats(mpm, nu)
    utau_ft  = utau_from_stats(ft,  nu)
    Retau_mpm = utau_mpm / nu   # = u_tau * H / nu  (H=1)
    Retau_ft  = utau_ft  / nu

    print(f"MPM-STD_C : u_tau = {utau_mpm:.5f}  Re_tau = {Retau_mpm:.1f}")
    print(f"Filt_TDMA : u_tau = {utau_ft :.5f}  Re_tau = {Retau_ft :.1f}")

    # Inner-layer scaling
    zp_mpm = mpm["zp"]
    zp_ft  = ft["zp"]

    Up_mpm  = mpm["Umean"] / utau_mpm
    Up_ft   = ft["Umean"]  / utau_ft

    urms_mpm = mpm["urms"] / utau_mpm
    urms_ft  = ft["urms"]  / utau_ft
    vrms_mpm = mpm["vrms"] / utau_mpm
    vrms_ft  = ft["vrms"]  / utau_ft
    wrms_mpm = mpm["wrms"] / utau_mpm
    wrms_ft  = ft["wrms"]  / utau_ft
    uw_mpm   = -mpm["uw"]  / utau_mpm**2
    uw_ft    = -ft["uw"]   / utau_ft**2

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Channel Re_tau≈180 — MPM-STD_C vs Filtered_TDMA", fontsize=13)

    kw_mpm = dict(color="tab:blue",   lw=2,   label="MPM-STD_C")
    kw_ft  = dict(color="tab:orange", lw=1.5, ls="--", label="Filt_TDMA")
    kw_log = dict(color="0.6",        lw=1,   ls=":",  label="log law: $\\kappa^{-1}\\ln z^+ + 5.5$")

    # ── U+ ──────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.semilogx(zp_mpm, Up_mpm, **kw_mpm)
    ax.semilogx(zp_ft,  Up_ft,  **kw_ft)
    zp_ref = np.logspace(np.log10(0.1), np.log10(max(zp_mpm.max(), zp_ft.max())), 200)
    ax.semilogx(zp_ref, np.log(zp_ref)/0.41 + 5.5, **kw_log)
    ax.semilogx(zp_ref, zp_ref, color="0.6", lw=1, ls="-.", label="$z^+$ (viscous sub-layer)")
    ax.set_xlabel("$z^+$"); ax.set_ylabel("$U^+$"); ax.set_title("Mean streamwise velocity")
    ax.set_xlim(0.1, max(zp_mpm.max(), zp_ft.max()) * 1.1)
    ax.legend(fontsize=8)

    # ── u_rms+ ──────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(zp_mpm, urms_mpm, **kw_mpm)
    ax.plot(zp_ft,  urms_ft,  **kw_ft)
    ax.set_xlabel("$z^+$"); ax.set_ylabel("$u'^+$"); ax.set_title("Streamwise RMS")
    ax.legend(fontsize=8)

    # ── v_rms+ ──────────────────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(zp_mpm, vrms_mpm, **kw_mpm)
    ax.plot(zp_ft,  vrms_ft,  **kw_ft)
    ax.set_xlabel("$z^+$"); ax.set_ylabel("$v'^+$"); ax.set_title("Spanwise RMS")
    ax.legend(fontsize=8)

    # ── w_rms+ ──────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(zp_mpm, wrms_mpm, **kw_mpm)
    ax.plot(zp_ft,  wrms_ft,  **kw_ft)
    ax.set_xlabel("$z^+$"); ax.set_ylabel("$w'^+$"); ax.set_title("Wall-normal RMS")
    ax.legend(fontsize=8)

    # ── -<uw>+ ──────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(zp_mpm, uw_mpm, **kw_mpm)
    ax.plot(zp_ft,  uw_ft,  **kw_ft)
    ax.set_xlabel("$z^+$"); ax.set_ylabel("$-\\langle uw \\rangle^+$"); ax.set_title("Reynolds shear stress")
    ax.legend(fontsize=8)

    # ── L∞ / L2 error table (text) ──────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    # Interpolate MPM onto FT z+ grid for error norms
    from numpy import interp
    Up_mpm_i  = interp(zp_ft, zp_mpm, Up_mpm)
    err_U  = np.abs(Up_mpm_i  - Up_ft)
    err_ur = np.abs(interp(zp_ft, zp_mpm, urms_mpm) - urms_ft)
    err_vr = np.abs(interp(zp_ft, zp_mpm, vrms_mpm) - vrms_ft)
    err_wr = np.abs(interp(zp_ft, zp_mpm, wrms_mpm) - wrms_ft)

    lines = [
        f"u_tau  MPM={utau_mpm:.5f}  FT={utau_ft:.5f}",
        f"Re_tau MPM={Retau_mpm:.1f}   FT={Retau_ft:.1f}",
        "",
        "L∞ error (MPM vs FT, inner units):",
        f"  U+    : {err_U.max():.3e}",
        f"  u_rms+: {err_ur.max():.3e}",
        f"  v_rms+: {err_vr.max():.3e}",
        f"  w_rms+: {err_wr.max():.3e}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
