#!/usr/bin/env python3
"""
convert_ft_to_mpm.py — Convert Filtered_TDMA restart files to MPM-STD_C format.

Layout differences
──────────────────
Filtered_TDMA  :  flat[i + Nx*(j + Ny*k)]  where i=x fastest, k=z slowest
                  file = (Nz, Ny, Nx) reshaped array
MPM-STD_C      :  flat[i*Ny*Nz + j*Nz + k] where k=z fastest, i=x slowest
                  file = (Nx, Ny, Nz) C-order array  (MPI subarray, MPI_ORDER_C)

Conversion: numpy.transpose(2, 1, 0) on the (Nz, Ny, Nx) array → (Nx, Ny, Nz).

Usage
─────
  python3 convert_ft_to_mpm.py \
      --ft_dir  /path/to/Filtered_TDMA/channel/restart_out/p1/ \
      --mpm_dir /path/to/MPM-STD_C/run/restart_Re180/ \
      --n1m 256 --n2m 256 --n3m 256

The script writes:
  <mpm_dir>/U.bin   V.bin   W.bin   P.bin    (raw double, MPM-STD_C layout)
  <mpm_dir>/meta.txt                          (step time dt dPdx)
"""

import argparse
import os
import struct
import sys

import numpy as np


def convert_field(ft_path: str, mpm_path: str, Nx: int, Ny: int, Nz: int,
                  field_name: str) -> None:
    size = Nx * Ny * Nz
    print(f"  {field_name}: reading  {ft_path}  ({size} doubles = {size*8//1024//1024} MB)")
    raw = np.fromfile(ft_path, dtype=np.float64)
    if raw.size != size:
        raise ValueError(f"{field_name}: expected {size} elements, got {raw.size}")

    # Filtered_TDMA layout: flat[i + Nx*(j + Ny*k)] → reshape as (Nz, Ny, Nx)
    arr_ft = raw.reshape(Nz, Ny, Nx)          # (z, y, x) axes

    # MPM-STD_C layout: C order (Nx, Ny, Nz) → transpose (z,y,x) → (x,y,z)
    arr_mpm = np.ascontiguousarray(arr_ft.transpose(2, 1, 0), dtype=np.float64)

    print(f"  {field_name}: writing  {mpm_path}")
    arr_mpm.tofile(mpm_path)
    print(f"  {field_name}: done  |min|={arr_mpm.min():.3e}  |max|={arr_mpm.max():.3e}")


def main():
    p = argparse.ArgumentParser(description="Convert Filtered_TDMA restart → MPM-STD_C")
    p.add_argument("--ft_dir",  required=True, help="Filtered_TDMA restart_out directory")
    p.add_argument("--mpm_dir", required=True, help="Output directory for MPM-STD_C restart")
    p.add_argument("--n1m", type=int, default=256, help="x cells")
    p.add_argument("--n2m", type=int, default=256, help="y cells")
    p.add_argument("--n3m", type=int, default=256, help="z cells")
    args = p.parse_args()

    os.makedirs(args.mpm_dir, exist_ok=True)
    Nx, Ny, Nz = args.n1m, args.n2m, args.n3m

    for fld in ("U", "V", "W", "P"):
        ft_path  = os.path.join(args.ft_dir,  f"cont_{fld}.bin")
        mpm_path = os.path.join(args.mpm_dir, f"{fld}.bin")
        convert_field(ft_path, mpm_path, Nx, Ny, Nz, fld)

    # Read cont_time.bin: (time, dt, step, dPdx) — 4 doubles
    time_path = os.path.join(args.ft_dir, "cont_time.bin")
    with open(time_path, "rb") as f:
        t_time, t_dt, t_step, t_dPdx = struct.unpack("4d", f.read(32))
    t_step = int(t_step)
    print(f"\n  time metadata: step={t_step}  time={t_time:.4f}  dt={t_dt:.6f}  dPdx={t_dPdx:.6e}")

    # Write MPM-STD_C meta.txt: "step time dt dPdx"
    meta_path = os.path.join(args.mpm_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"{t_step} {t_time} {t_dt} {t_dPdx}\n")
    print(f"  meta.txt written → {meta_path}")
    print("\nConversion complete.")


if __name__ == "__main__":
    main()
