#!/usr/bin/env python3
"""eoc_dhvc.py — Mesh-refinement convergence study for DHVC.

Runs MPM-STD_C and PaScaL_TCS at n_wall = 8, 16, 32, 64 and prints
L∞(U - U_ana) and EOC for each code.

Settings match the original PaScaL_TCS DHVC benchmark:
  Ra=100, Pr=1, tanh-stretched wall-normal grid (gamma=2.6),
  domain 4H × H × 2H, N_XY=32 stream/span cells.

PaScaL reaches steady-state at n=8 and n=16 within 2000 time steps.
At n=32 and n=64 the NOB transient forces tiny dt; only MPM results
are meaningful there.

Usage:
  python3 eoc_dhvc.py
"""

import sys, os, math, struct, subprocess, shutil, re, tempfile

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MPM_ROOT     = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
PASCAL_ROOT  = os.path.abspath(os.path.join(MPM_ROOT, "../PaScaL_TCS"))
PASCAL_RUN   = os.path.join(PASCAL_ROOT, "run")
MPM_BIN      = os.path.join(MPM_ROOT, "build/cpu/bin/rbc")
PASCAL_BIN   = os.path.join(PASCAL_RUN, "PaScaL_TCS.ex")
MPM_TOML     = os.path.join(MPM_ROOT, "apps/rbc/input.toml")

# NVIDIA mpirun needed for PaScaL (compiled with HPC-SDK Fortran)
NVMPI = "/opt/nvidia/hpc-sdk/Linux_x86_64/23.7/comm_libs/openmpi/openmpi-3.1.5/bin/mpirun"
SYS_MPI = "mpirun"

# ── physics ────────────────────────────────────────────────────────────────────
Ra = 100.0;  Pr = 1.0;  nu = math.sqrt(Pr / Ra);  Lz = 1.0
N_XY  = 32      # stream / span (matches original PaScaL benchmark)
GAMMA = 2.6     # tanh stretching (wall-normal)

N_WALL = [64, 128, 256, 512]

# ── grid ───────────────────────────────────────────────────────────────────────
def tanh_faces(n, L, g):
    hg = 0.5 * g
    d  = math.tanh(hg)
    return [0.5*L*(1 + math.tanh(hg*(2*i/n - 1))/d) for i in range(n+1)]

def cell_centers(n):
    zf = tanh_faces(n, Lz, GAMMA)
    return [0.5*(zf[k]+zf[k+1]) for k in range(n)]

def U_ana(z):
    return z*(2*z/Lz - 1)*(z/Lz - 1) / (12*nu)

# ── I/O ────────────────────────────────────────────────────────────────────────
def load_f64(path):
    with open(path, "rb") as f: data = f.read()
    return list(struct.unpack(f"<{len(data)//8}d", data))

def z_profile(flat, n1, n2, n3):
    """Average over stream(n1) and span(n2); layout [stream][wall][span]."""
    prof = [0.0]*n3
    for ii in range(n1):
        for kk in range(n3):
            for jj in range(n2):
                prof[kk] += flat[ii*n3*n2 + kk*n2 + jj]
    return [v/(n1*n2) for v in prof]

def linf_U_err(U_num, zc):
    return max(abs(U_num[k] - U_ana(zc[k])) for k in range(len(zc)))

# ── MPM runner ─────────────────────────────────────────────────────────────────
def run_mpm(n_wall):
    with open(MPM_TOML) as f: toml = f.read()
    toml = re.sub(r'(?m)^n3m\s*=\s*\d+', f'n3m = {n_wall}', toml)
    toml = re.sub(r'(?m)^t_end\s*=\s*[\d.]+', 't_end = 60.0', toml)
    toml = re.sub(r'(?m)^n_steps\s*=\s*\d+', 'n_steps = 100000', toml)
    toml = re.sub(r'(?m)^u_ic_dhvc\s*=\s*\d+', 'u_ic_dhvc = 1', toml)

    with tempfile.TemporaryDirectory() as tmp:
        tp = os.path.join(tmp, "input.toml")
        with open(tp, "w") as f: f.write(toml)

        r = subprocess.run([SYS_MPI, "-np", "1", MPM_BIN, tp],
                           capture_output=True, text=True, cwd=tmp)
        if r.returncode != 0:
            print(f"  [MPM n={n_wall}] FAIL\n{r.stderr[-300:]}")
            return None
        flat = load_f64(os.path.join(tmp, "dump_mpm_U.bin"))

    zc = cell_centers(n_wall)
    # MPM dump layout: [stream(n1m)][wall(n3m)][span(n2m)]
    return linf_U_err(z_profile(flat, N_XY, N_XY, n_wall), zc)

# ── PaScaL runner ──────────────────────────────────────────────────────────────
PARA_TMPL = """\
&sim_continue
ContinueFilein      = 0
ContinueFileout     = 0
dir_cont_filein     = './data/1_continue/MPI_IO/'
dir_cont_fileout    = './data/1_continue/'
dir_instantfield    = './data/2_instanfield/'
/

&meshes
n1m = {n1m}
n2m = {n2m}
n3m = {n3m}
/

&MPI_procs
np1 = 1
np2 = 1
np3 = 1
/

&periodic_boundary
pbc1 = .true.
pbc2 = .false.
pbc3 = .true.
/

&uniform_mesh
uniform1 = 1
uniform2 = 0
uniform3 = 1
/

&mesh_stretch
gamma1 = 3.0
gamma2 = 2.6d0
gamma3 = 3.0
/

&aspect_ratio
Aspect1 = 4
H       = 1
Aspect3 = 2
/

&sim_parameter
Ra      = 1E2
Pr      = 1
DeltaT  = 1.
MaxCFL  = 0.5
/

&sim_control
dtStart             = 5.0D-2
tStart              = 0.d0
Timestepmax         = 2000
print_start_step    = 1
print_interval_step = 500
dtMax               = 1.0D0
/
"""

def run_pascal(n_wall):
    """Run PaScaL with n2m=n_wall (wall-normal). PaScaL writes dumps to run/."""
    para_path = os.path.join(PASCAL_RUN, "PARA_INPUT.dat")
    para_bak  = para_path + ".bak"
    shutil.copy2(para_path, para_bak)

    para = PARA_TMPL.format(n1m=N_XY, n2m=n_wall, n3m=N_XY)
    with open(para_path, "w") as f: f.write(para)

    for d in ["data/1_continue", "data/2_instanfield"]:
        os.makedirs(os.path.join(PASCAL_RUN, d), exist_ok=True)

    try:
        r = subprocess.run([NVMPI, "-np", "1", "./PaScaL_TCS.ex"],
                           capture_output=True, text=True, cwd=PASCAL_RUN)
        if r.returncode != 0:
            print(f"  [PaScaL n={n_wall}] FAIL\n{r.stderr[-300:]}")
            return None

        ubin = os.path.join(PASCAL_RUN, "dump_pascal_U.bin")
        if not os.path.exists(ubin):
            print(f"  [PaScaL n={n_wall}] dump_pascal_U.bin missing")
            return None

        flat = load_f64(ubin)
    finally:
        shutil.copy2(para_bak, para_path)

    zc = cell_centers(n_wall)
    # PaScaL dump layout: outer=stream(n1m), middle=wall(n2m), inner=span(n3m)
    return linf_U_err(z_profile(flat, N_XY, N_XY, n_wall), zc)

# ── EOC ────────────────────────────────────────────────────────────────────────
def eoc(e_c, e_f, n_c, n_f):
    if None in (e_c, e_f) or e_f <= 0: return float("nan")
    return math.log(e_c/e_f) / math.log(n_f/n_c)

# ── main ────────────────────────────────────────────────────────────────────────
def main():
    mpm_e = {}; pas_e = {}

    for n in N_WALL:
        print(f"\n── n_wall = {n} ──────────────────────────────────────")
        sys.stdout.flush()
        print(f"  MPM  ...", end=" ", flush=True)
        mpm_e[n] = run_mpm(n)
        print(f"L∞(U) = {mpm_e[n]:.3e}" if mpm_e[n] else "FAILED")

        print(f"  PaScaL ...", end=" ", flush=True)
        pas_e[n] = run_pascal(n)
        print(f"L∞(U) = {pas_e[n]:.3e}" if pas_e[n] else "FAILED")

    print("\n" + "="*70)
    print(f"{'n_wall':>8}  {'MPM L∞(U)':>12}  {'MPM EOC':>8}  "
          f"{'PaScaL L∞(U)':>14}  {'PaScaL EOC':>10}")
    print("-"*70)
    prev = None
    for n in N_WALL:
        me = mpm_e[n]; pe = pas_e[n]
        m_eoc = eoc(mpm_e[prev], me, prev, n) if prev else float("nan")
        p_eoc = eoc(pas_e[prev], pe, prev, n) if prev else float("nan")
        ms = f"{me:.3e}" if me else " FAIL"
        ps = f"{pe:.3e}" if pe else " FAIL"
        m_eoc_s = f"{m_eoc:.2f}" if not math.isnan(m_eoc) else "  —"
        p_eoc_s = f"{p_eoc:.2f}" if not math.isnan(p_eoc) else "  —"
        print(f"{n:>8}  {ms:>12}  {m_eoc_s:>8}  {ps:>14}  {p_eoc_s:>10}")
        prev = n
    print("="*70)

if __name__ == "__main__":
    main()
