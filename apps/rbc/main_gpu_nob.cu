// RBC — GPU backend (G3b hybrid), NOB (variable properties: glycerol).
// Scalar + momentum + properties solved on GPU; pressure solved on CPU (hybrid G3b).
// Gravity along z. Hot bottom +0.5, cold top -0.5. No-slip U,V,W.

#include "rbc_common.hpp"
#include "core/gpu/gpu_grid.hpp"
#include <chrono>
#include <cuda_runtime.h>

namespace {
using hrc = std::chrono::high_resolution_clock;
inline double ms_since(hrc::time_point t0) {
  return std::chrono::duration<double,std::milli>(hrc::now()-t0).count();
}

// GPU kernel: zero W at wall faces (replaces D2H+CPU zero_w_wall+H2D each step)
__global__ void k_zero_w_wall_gpu(double* W, int n2t, int n3t, int h,
                                   int n1t, bool has_lower, bool has_upper) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n1t || j >= n2t) return;
  if (has_lower) W[(i*n2t+j)*n3t + h]        = 0.0;
  if (has_upper) W[(i*n2t+j)*n3t + (n3t - h)] = 0.0;
}

// GPU kernel: clamp T to [-0.5, 0.5] on INTERIOR cells only. The z-wall ghost
// planes hold the antisymmetric Dirichlet mirror 2*T_wall - T_in, which
// legitimately exceeds the physical range (0.5155 at the hot wall). Clipping
// ghosts pinned the effective wall at the ghost CENTER (half a cell outside
// the face) and halved the wall conductance every step — measured as the 2x
// wall-gap flux kink in the mean T profile (Report/31).
// clip_count: running counter of clipped cells (audit D3, Report/RBC — the
// clip is not in the paper and acts as a non-conservative heat sink/source;
// if it fires at statistical steady state it can bias theta_c, so count it).
__global__ void k_clip_T_gpu(double* T, int n1i, int n2i, int n3i,
                             int n2t, int n3t, unsigned long long* clip_count) {
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.z * blockDim.z + threadIdx.z;
  if (ii >= n1i || jj >= n2i || kk >= n3i) return;
  const std::size_t c = (((std::size_t)(ii + kHaloWidth)) * n2t + (jj + kHaloWidth)) * n3t
                      + (kk + kHaloWidth);
  const double v = T[c];
  if (v < -0.5 || v > 0.5) {
    T[c] = v < -0.5 ? -0.5 : 0.5;
    atomicAdd(clip_count, 1ULL);
  }
}

// ── GPU statistics kernels — eliminate per-step 64 MB D2H ────────────────────

__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
  unsigned long long* ull = (unsigned long long*)addr;
  unsigned long long old = *ull, assumed;
  do { assumed = old;
       old = atomicCAS(ull, assumed,
           __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
}

// Compute tc_local (∑T at kc_local z-plane) and wt_local (∑W*T*dz) for this rank.
// d_out[0]=wt, d_out[1]=tc  — must be zeroed before call.
__global__ void k_accum_tc_wt_rbc(
    const double* T, const double* W, const double* dx3,
    double* d_out,
    int nx, int ny, int nz, int n2t, int n3t, int h, int kc_local)
{
  extern __shared__ double shm[];
  const int ii = blockIdx.x*blockDim.x+threadIdx.x;
  const int jj = blockIdx.y*blockDim.y+threadIdx.y;
  const int kk = blockIdx.z*blockDim.z+threadIdx.z;
  const int tid = (threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
  const int bsz = blockDim.x*blockDim.y*blockDim.z;
  double wt=0.0, tc=0.0;
  if (ii<nx && jj<ny && kk<nz) {
    const int i=ii+h, j=jj+h, k=kk+h;
    const double tv = T[(std::size_t)(i*n2t+j)*n3t+k];
    const double wv = W[(std::size_t)(i*n2t+j)*n3t+k];
    wt = wv * tv * dx3[k];
    if (kk == kc_local) tc = tv;
  }
  shm[tid]=wt; shm[bsz+tid]=tc;
  __syncthreads();
  for (int s=bsz/2; s>0; s>>=1) {
    if (tid<s) { shm[tid]+=shm[tid+s]; shm[bsz+tid]+=shm[bsz+tid+s]; }
    __syncthreads();
  }
  if (tid==0) { atomicAdd(d_out+0, shm[0]); atomicAdd(d_out+1, shm[bsz]); }
}

// Accumulate z-profiles: d_t_sum_z[kk]+=∑_{i,j}T[i,j,kk], d_w2_sum_z[kk]+=∑W².
// Grid: (ceil(nx*ny/256), nz) blocks, 256 threads, 512*sizeof(double) shared mem.
__global__ void k_accum_z_profile_rbc(
    const double* T, const double* W,
    double* d_t_sum_z, double* d_w2_sum_z,
    int nx, int ny, int nz, int n2t, int n3t, int h)
{
  extern __shared__ double shm[];
  const int kk = blockIdx.y;
  const int flat = blockIdx.x*blockDim.x+threadIdx.x;
  const int tid = threadIdx.x, bsz = blockDim.x;
  double tv=0.0, w2v=0.0;
  if (flat<nx*ny && kk<nz) {
    const int ii=flat/ny, jj=flat%ny, i=ii+h, j=jj+h, k=kk+h;
    tv = T[(std::size_t)(i*n2t+j)*n3t+k];
    const double wv = W[(std::size_t)(i*n2t+j)*n3t+k]; w2v = wv*wv;
  }
  shm[tid]=tv; shm[bsz+tid]=w2v;
  __syncthreads();
  for (int s=bsz/2; s>0; s>>=1) {
    if (tid<s) { shm[tid]+=shm[tid+s]; shm[bsz+tid]+=shm[bsz+tid+s]; }
    __syncthreads();
  }
  if (tid==0 && kk<nz) { atomicAdd(&d_t_sum_z[kk], shm[0]); atomicAdd(&d_w2_sum_z[kk], shm[bsz]); }
}

// Compute umax, vmax, wmax, Tmax, -Tmin  (all as max reductions).
// d_out[5] — must be zeroed before call.
__global__ void k_flow_diag_rbc(
    const double* U, const double* V, const double* W, const double* T,
    double* d_out, int nx, int ny, int nz, int n2t, int n3t, int h)
{
  const int ii=blockIdx.x*blockDim.x+threadIdx.x;
  const int jj=blockIdx.y*blockDim.y+threadIdx.y;
  const int kk=blockIdx.z*blockDim.z+threadIdx.z;
  if (ii>=nx||jj>=ny||kk>=nz) return;
  const std::size_t c = (std::size_t)((ii+h)*n2t+(jj+h))*n3t+(kk+h);
  atomicMaxDouble(d_out+0, fabs(U[c]));
  atomicMaxDouble(d_out+1, fabs(V[c]));
  atomicMaxDouble(d_out+2, fabs(W[c]));
  atomicMaxDouble(d_out+3, T[c]);
  atomicMaxDouble(d_out+4, -T[c]);
}

// Compute max |∇·U|.  d_out[0] — must be zeroed before call.
__global__ void k_div_max_rbc(
    const double* U, const double* V, const double* W,
    const double* dx1, const double* dx2, const double* dx3,
    double* d_out, int nx, int ny, int nz, int n2t, int n3t, int h)
{
  const int ii=blockIdx.x*blockDim.x+threadIdx.x;
  const int jj=blockIdx.y*blockDim.y+threadIdx.y;
  const int kk=blockIdx.z*blockDim.z+threadIdx.z;
  if (ii>=nx||jj>=ny||kk>=nz) return;
  const int i=ii+h, j=jj+h, k=kk+h;
  const double dv =
    (U[(std::size_t)((i+1)*n2t+j)*n3t+k] - U[(std::size_t)(i*n2t+j)*n3t+k]) / dx1[i] +
    (V[(std::size_t)(i*n2t+j+1)*n3t+k]   - V[(std::size_t)(i*n2t+j)*n3t+k]) / dx2[j] +
    (W[(std::size_t)(i*n2t+j)*n3t+k+1]   - W[(std::size_t)(i*n2t+j)*n3t+k]) / dx3[k];
  atomicMaxDouble(d_out, fabs(dv));
}

// ── GpuRbcAccum: device-side stats buffers + GPU accumulate/print helpers ────
struct GpuRbcAccum {
  double* d_tmp2    = nullptr;   // [2]  scratch: {wt_local, tc_local}
  double* d_diag6   = nullptr;   // [6]  scratch: {umax,vmax,wmax,Tmax,-Tmin,div_max}
  double* d_t_sum_z = nullptr;   // [nzl] z-profile of T (accumulated each step)
  double* d_w2_sum_z= nullptr;   // [nzl] z-profile of W² (accumulated each step)
  int nzl, nx, ny, nz, n2t, n3t, h, kc_local, nxg, nyg, koff, nz_global;
  const double* d_dx1; const double* d_dx2; const double* d_dx3;

  GpuRbcAccum(const core::Subdomain& sub, const core::gpu::GpuGridArrays& gg) {
    const auto ni = sub.n_interior(); nx=ni[0]; ny=ni[1]; nz=ni[2]; nzl=nz;
    const auto nt = sub.n_total();    n2t=nt[1]; n3t=nt[2]; h=kHaloWidth;
    const auto ng = sub.n_global();   nxg=ng[0]; nyg=ng[1]; nz_global=ng[2];
    const int nzg = ng[2], kc_g = nzg/2;
    koff = sub.global_offset()[2];
    kc_local = (kc_g>=koff && kc_g<koff+nzl) ? (kc_g - koff) : -1;
    cudaMalloc(&d_tmp2,     2*sizeof(double));
    cudaMalloc(&d_diag6,    6*sizeof(double));
    cudaMalloc(&d_t_sum_z,  nzl*sizeof(double));
    cudaMalloc(&d_w2_sum_z, nzl*sizeof(double));
    cudaMemset(d_t_sum_z,  0, nzl*sizeof(double));
    cudaMemset(d_w2_sum_z, 0, nzl*sizeof(double));
    d_dx1=gg.d_dx1(); d_dx2=gg.d_dx2(); d_dx3=gg.d_dx3();
  }
  ~GpuRbcAccum() {
    cudaFree(d_tmp2); cudaFree(d_diag6);
    cudaFree(d_t_sum_z); cudaFree(d_w2_sum_z);
  }
};

// Raw per-step (unaccumulated) tc/wt reduction — shared by gpu_accumulate_rbc
// (which folds this step's contribution into the long-time running sum) and
// gpu_print_step_rbc (which now reports THIS step's instant value directly,
// instead of the cumulative running mean, so transients are visible live
// without having to back out local windows from the cumulative print by hand).
inline void gpu_compute_instant_tc_wt(GpuRbcAccum& ga, const core::GpuFields& gf,
    MPI_Comm comm, double& wt_instant, double& tc_instant)
{
  const double* T = gf[core::Var::T].data();
  const double* W = gf[core::Var::W].data();
  cudaMemset(ga.d_tmp2, 0, 2*sizeof(double));
  const dim3 blk(8,8,4);
  const dim3 grd((ga.nx+7)/8,(ga.ny+7)/8,(ga.nz+3)/4);
  k_accum_tc_wt_rbc<<<grd,blk,2*256*sizeof(double)>>>(
      T, W, ga.d_dx3, ga.d_tmp2,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h, ga.kc_local);
  cudaDeviceSynchronize();
  double local[2]; cudaMemcpy(local, ga.d_tmp2, 2*sizeof(double), cudaMemcpyDeviceToHost);
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, comm);
  wt_instant = global[0];
  tc_instant = global[1] / (double(ga.nxg)*ga.nyg);
}

// GPU accumulate: compute tc and wt on device, D2H 16 bytes, MPI_Allreduce → update stats.
inline void gpu_accumulate_rbc(GpuRbcAccum& ga, rbc::RbcStats& s,
    const core::GpuFields& gf, MPI_Comm comm)
{
  const double* T = gf[core::Var::T].data();
  const double* W = gf[core::Var::W].data();
  double wt_instant, tc_instant;
  gpu_compute_instant_tc_wt(ga, gf, comm, wt_instant, tc_instant);
  // k_accum_z_profile: 256 threads, grid (ceil(nx*ny/256), nz)
  const int flat_blocks = (ga.nx*ga.ny+255)/256;
  k_accum_z_profile_rbc<<<dim3(flat_blocks,ga.nz),256,2*256*sizeof(double)>>>(
      T, W, ga.d_t_sum_z, ga.d_w2_sum_z,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h);
  cudaDeviceSynchronize();
  s.wt_sum += wt_instant;
  s.tc_sum += tc_instant;
  ++s.tc_n;
}

// GPU print_step: flow_diag + div_max on device, D2H 48 bytes, MPI_Allreduce → printf + CSV.
// Total D2H per call: 48 bytes.
inline void gpu_print_step_rbc(GpuRbcAccum& ga, const rbc::RbcStats& s,
    const core::GpuFields& gf, int step, double time, real_t dt, double alpha_T,
    bool is_root, MPI_Comm comm, FILE* mon_fp)
{
  const double* U=gf[core::Var::U].data(); const double* V=gf[core::Var::V].data();
  const double* W=gf[core::Var::W].data(); const double* T=gf[core::Var::T].data();
  cudaMemset(ga.d_diag6, 0, 6*sizeof(double));
  const dim3 blk(8,8,4);
  const dim3 grd((ga.nx+7)/8,(ga.ny+7)/8,(ga.nz+3)/4);
  k_flow_diag_rbc<<<grd,blk>>>(U,V,W,T, ga.d_diag6,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h);
  k_div_max_rbc<<<grd,blk>>>(U,V,W, ga.d_dx1, ga.d_dx2, ga.d_dx3, ga.d_diag6+5,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h);
  cudaDeviceSynchronize();
  double local[6]; cudaMemcpy(local, ga.d_diag6, 6*sizeof(double), cudaMemcpyDeviceToHost);
  double global[6];
  MPI_Allreduce(local, global, 6, MPI_DOUBLE, MPI_MAX, comm);
  // Instant (this-step-only) Theta_c/Nu — NOT the cumulative running mean —
  // so transients/drift are visible live without hand-differencing the
  // cumulative print across monitor intervals. Computed every monitor call
  // regardless of nstat_start, so it's meaningful pre-stats-window too.
  double wt_instant, tc_instant;
  gpu_compute_instant_tc_wt(ga, gf, comm, wt_instant, tc_instant);
  if (!is_root) return;
  const double dv=global[5], umax=global[0], vmax=global[1], wmax=global[2];
  const double Tmax=global[3], Tmin=-global[4];
  const double tc = tc_instant;
  const double nu_val = 1.0 + (wt_instant/(double(ga.nxg)*ga.nyg))/alpha_T;
  std::printf("step=%d t=%.3g dt=%.3g div=%.2e Theta_c=%.4f | umax=%.3g vmax=%.3g wmax=%.3g T=[%.3f,%.3f]\n",
              step, time, double(dt), dv, tc, umax, vmax, wmax, Tmin, Tmax);
  std::printf("  [Nu] instant: Nu=%.4f\n", nu_val);
  if (mon_fp) {
    std::fprintf(mon_fp, "%d,%.6g,%.6g,%.6e,%.6f,%.6f\n",
                 step, time, double(dt), dv, tc, nu_val);
    std::fflush(mon_fp);
  }
}

// Write z-profile stats file: D2H ~nzl*16 B (t_sum + w2_sum), gather, write CSV.
inline void gpu_write_stats_rbc(GpuRbcAccum& ga, const rbc::RbcStats& s,
    const std::string& path, int step, double Ra, double DeltaT,
    bool is_root, MPI_Comm comm,
    const core::Grid& grid, const core::Subdomain& sub)
{
  if (s.tc_n == 0) return;
  const int nzl = ga.nzl, nzg = ga.nz_global;
  const double inv = 1.0 / (s.tc_n * static_cast<double>(ga.nxg) * ga.nyg);
  // D2H local z-profiles
  std::vector<double> t_loc(nzl), w2_loc(nzl);
  cudaMemcpy(t_loc.data(),  ga.d_t_sum_z,  nzl*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(w2_loc.data(), ga.d_w2_sum_z, nzl*sizeof(double), cudaMemcpyDeviceToHost);
  // Gather global profiles
  std::vector<double> t_send(nzg,0.0), w2_send(nzg,0.0);
  std::vector<double> t_glb(nzg,0.0),  w2_glb(nzg,0.0);
  for (int kl=0; kl<nzl; ++kl) {
    t_send[ga.koff+kl]  = t_loc[kl]  * inv;
    w2_send[ga.koff+kl] = w2_loc[kl] * inv;
  }
  MPI_Allreduce(t_send.data(),  t_glb.data(),  nzg, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(w2_send.data(), w2_glb.data(), nzg, MPI_DOUBLE, MPI_SUM, comm);
  if (!is_root) return;
  FILE* fp = std::fopen(path.c_str(), "w");
  if (!fp) { std::fprintf(stderr, "[rbc_stats] cannot open '%s'\n", path.c_str()); return; }
  std::fprintf(fp, "# RBC z-profile stats (step=%d, n=%ld, Ra=%.2e, DeltaT=%.1f)\n",
               step, s.tc_n, Ra, DeltaT);
  std::fprintf(fp, "# Z,T_mean,W_rms\n");
  // global offset per rank is baked into t_glb — just index by global k
  for (int k=0; k<nzg; ++k) {
    // zc is local; for rank 0 global offset=0, so zc[h+k] is valid only for k < nzl
    // Use the gathered global offset via sub.n_global()[2] and grid cell spacing
    // Safe: write index k and T_mean, W_rms
    const double w_rms = std::sqrt(std::max(w2_glb[k], 0.0));
    std::fprintf(fp, "%.8e,%.8e,%.8e\n",
                 static_cast<double>(k+1) / static_cast<double>(nzg),  // normalized z
                 t_glb[k], w_rms);
  }
  std::fclose(fp);
}
} // anonymous namespace


//----------------------------------------------- Main -----------------------------------------------
int main(int argc, char** argv) {
  core::MpiContext mpi(&argc, &argv);
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);

  auto is_per = [](const std::string& s){ return s == "periodic"; };
  std::array<int,3>  dims     = {cfg.get<int>("mpi","np1"), cfg.get<int>("mpi","np2"), cfg.get<int>("mpi","np3")};
  std::array<bool,3> periodic = {is_per(cfg.get_or<std::string>("topology","x","periodic")),
                                 is_per(cfg.get_or<std::string>("topology","y","periodic")),
                                 is_per(cfg.get_or<std::string>("topology","z","wall"))};
  core::MpiTopology topo(mpi, dims, periodic);
  std::array<int,3> n_global = {cfg.get<int>("mesh","n1m"), cfg.get<int>("mesh","n2m"), cfg.get<int>("mesh","n3m")};
  core::Subdomain sub(topo, n_global);

  std::array<grid::AxisConfig,3> axes;
  for (int a = 0; a < 3; ++a) {
    const int    is_uni = cfg.get_or<int>   ("uniform_mesh","uniform"+std::to_string(a+1), 1);
    const double gamma  = cfg.get_or<double>("mesh_stretch","gamma" +std::to_string(a+1), 0.0);
    const double L      = cfg.get<double>   ("domain", std::string("L")+char('x'+a));
    axes[a] = {n_global[a], L, is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
  }
  core::Grid grid(sub, axes);
  core::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
  bc.validate();
  auto  tdma_owner = linear_solver::tdma::TdmaRegistry::make_default(topo);
  core::Domain domain{grid, topo, sub, *tdma_owner};

  core::CpuFields fields;
  fields.add(core::Var::U, sub); fields.add(core::Var::V, sub);
  fields.add(core::Var::W, sub); fields.add(core::Var::P, sub);
  fields.add(core::Var::T, sub);

  const double Ra = cfg.get<double>("physics","Ra"), Pr = cfg.get<double>("physics","Pr");
  const double nu = std::sqrt(Pr/Ra), alpha_T = 1.0/std::sqrt(Ra*Pr);
  fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  core::PressureSystem pressure;

  const double DeltaT = cfg.get_or<double>("physics","DeltaT", 0.0);
  physics::CpuPropertyFields props(sub);   // CPU copy for pressure step
  const physics::PropertyModel pmodel = physics::PropertyModel::glycerol(static_cast<real_t>(DeltaT));
  physics::BuoyancyParams buoy;
  buoy.gravity_axis = cfg.get_or<int>("physics","gravity_axis", 2); buoy.coeff = 1; buoy.T_ref = 0;
  buoy.a12pera11 = (std::fabs(pmodel.a[2]) < real_t{1e-14}) ? real_t{0} : pmodel.a[2] / pmodel.a[1];
  buoy.delta_T   = static_cast<real_t>(DeltaT);

  const double t_end    = cfg.get_or<double>("time","t_end",   400.0);
  const real_t dt_cap   = static_cast<real_t>(cfg.get_or<double>("time","dt_start",  0.02));
  const real_t max_cfl  = static_cast<real_t>(cfg.get_or<double>("time","MaxCFL",    0.5));
  const int    n_steps  = cfg.get_or<int>   ("time","n_steps", 200000);
  const int    nmonitor    = cfg.get_or<int>("output","nmonitor",    200);
  const int    nstat_start = cfg.get_or<int>("output","nstat_start", 20000);
  const int    nstat_write = cfg.get_or<int>("output","nstat_write", 10000);
  const std::string dir_stats   = cfg.get_or<std::string>("output","dir_stats","apps/rbc/statistics/");
  const std::string restart_in  = cfg.get_or<std::string>("restart","dir_in",  "");
  const std::string restart_out = cfg.get_or<std::string>("restart","dir_out", "");
  
  // Periodic checkpoint interval in steps (0 = off, final-only write as before).
  // Checkpoints alternate between <dir_out>/ckpt_a and <dir_out>/ckpt_b so a job
  // killed mid-write (timeout, node failure) never corrupts the only copy; to
  // resume, point [restart] dir_in at the ckpt dir whose stats.txt has the
  // larger step. Motivated by P5/job 30282: 48 h TimeLimit < 60 h cold run,
  // restart was final-only → the whole equilibrated field would have been lost.
  const int nrestart_write = cfg.get_or<int>("restart","nrestart_write", 0);
  if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
  MPI_Barrier(topo.cart_comm());

  // IC: linear T(z) + multi-mode W perturbation, or restart from file
  rbc::RestartMeta rmeta;
  if (restart_in.empty()) {
    const double Lx=cfg.get<double>("domain","Lx"), Ly=cfg.get<double>("domain","Ly"), Lz=cfg.get<double>("domain","Lz");
    const double T_lo=bc.T.face(Direction::Z,Side::Minus).value, T_hi=bc.T.face(Direction::Z,Side::Plus).value;
    const double amp=cfg.get_or<double>("init","perturb",0.02);
    const auto& xc=grid.xc(Direction::X); const auto& yc=grid.xc(Direction::Y); const auto& zc=grid.xc(Direction::Z);
    const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2];
    real_t* W=fields[core::Var::W].data(); real_t* T=fields[core::Var::T].data();
    for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) for (int k=0;k<n3;++k) {
      const int c=(i*n2+j)*n3+k;
      const double z=zc[k], x=xc[i], y=yc[j];
      T[c]=static_cast<real_t>(T_lo+(T_hi-T_lo)*(z/Lz));
      const double env=std::sin(M_PI*z/Lz);
      W[c]+=static_cast<real_t>(amp*env*(std::sin(2*M_PI*x/Lx)*std::cos(2*M_PI*y/Ly)
                                        +std::cos(2*M_PI*x/Lx)*std::sin(2*M_PI*y/Ly)
                                        +std::sin(4*M_PI*x/Lx)*std::sin(4*M_PI*y/Ly)));
    }
    rbc::zero_w_wall(fields[core::Var::W], sub);
    core::sync_field_cpu(fields, core::Var::U, bc, sub); core::sync_field_cpu(fields, core::Var::V, bc, sub);
    core::sync_field_cpu(fields, core::Var::W, bc, sub); core::sync_field_cpu(fields, core::Var::P, bc, sub);
    core::sync_field_cpu(fields, core::Var::T, bc, sub);
  } else {
    rmeta = rbc::read_restart(fields, bc, sub, restart_in, topo.cart_comm());
    core::sync_field_cpu(fields, core::Var::P, bc, sub);
  }

  // GPU setup: bind device, upload grid metrics, allocate GPU fields
  core::bind_gpu_to_local_rank_gpu(mpi);
  core::gpu::GpuGridArrays gpu_grid_arrays(grid);
  domain.gpu_grid = &gpu_grid_arrays;

  core::GpuFields gpu_fields;
  gpu_fields.add(core::Var::U, sub); gpu_fields.add(core::Var::V, sub);
  gpu_fields.add(core::Var::W, sub); gpu_fields.add(core::Var::P, sub);
  gpu_fields.add(core::Var::T, sub);
  gpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
  gpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

  core::GpuMomentumSystem  gpu_momentum(sub);
  core::GpuScalarSystem    gpu_scalar(sub);
  physics::GpuPropertyFields gpu_props(sub);

  gpu_fields[core::Var::U].to_device(fields[core::Var::U].data());
  gpu_fields[core::Var::V].to_device(fields[core::Var::V].data());
  gpu_fields[core::Var::W].to_device(fields[core::Var::W].data());
  gpu_fields[core::Var::P].to_device(fields[core::Var::P].data());
  gpu_fields[core::Var::T].to_device(fields[core::Var::T].data());

  // Incremental (van Kan) projection: persistent accumulated pressure Pⁿ.
  // Predictor subtracts dt·(1/ρ̂)∇Pⁿ; the Poisson then solves the increment δp
  // (Var::P) and Pⁿ⁺¹ = Pⁿ + δp. Seeded from the restart P (zero when absent —
  // self-starting: the first solve returns the full pressure). Report/32.
  core::GpuField gpu_P_acc(sub, "P_acc");
  gpu_P_acc.to_device(fields[core::Var::P].data());

  if (mpi.is_root())
    std::printf("[rbc_gpu_nob] Ra=%.2e Pr=%.4g DeltaT=%.1f nu=%.4e alpha=%.4e  n=%dx%dx%d  (GPU hybrid)\n",
                Ra, Pr, DeltaT, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

  // ── time loop ─────────────────────────────────────────────────────────────
  double time = rmeta.time; int step = rmeta.step;
  rbc::RbcStats stats(sub.n_global()[2]);
  // stats_reset=1: start the running mean completely fresh from this restart
  // point (ignore the carried-over tc_sum/tc_n/wt_sum) instead of diluting a new
  // accumulation window with old history — see Report/35's "cumulative mean
  // dilutes/lags the true trend" finding.
  const bool stats_reset = cfg.get_or<int>("restart", "stats_reset", 0) != 0;
  if (!stats_reset) rbc::apply_stats_from_restart(stats, rmeta);
  else if (mpi.is_root())
    std::printf("[restart] stats_reset=1 — starting Theta_c/Nu accumulation fresh from step=%d\n", step);
  const real_t* dx3_z = grid.dx_ptr(Direction::Z);
  GpuRbcAccum gpu_accum(sub, gpu_grid_arrays);

  FILE* mon_fp = nullptr;
  if (mpi.is_root()) {
    const std::string mon_path = dir_stats + "monitor.csv";
    mon_fp = std::fopen(mon_path.c_str(), "w");
    if (mon_fp) std::fprintf(mon_fp, "step,t,dt,div,Theta_c,Nu\n");
  }

  // Per-section timing accumulators (only rank 0 prints)
  double t_scalar=0, t_mom=0, t_zero_w=0, t_pressure=0, t_clip=0, t_diag=0, t_accum=0;
  const int n_time_steps = 10; // profile first N steps

  unsigned long long* d_clip_count = nullptr;   // audit D3: clip activation counter

  // ── 한 스텝의 흐름 ──────────────────────────────────────────────────────────
  //  [1] CFL → dt   [2] 물성 갱신(Tⁿ)   [3] 에너지 방정식 → Tⁿ⁺¹
  //  [4] 물성 재갱신(Tⁿ⁺¹)   [5] 운동량 예측자 → U*,V*,W*   [6] 벽면 W=0
  //  [7] 압력 Poisson + 투영 → 발산 제거, Pⁿ⁺¹=Pⁿ+δp   [8] T 클리핑(NOB)
  //  [9] 통계 누적 / 모니터 출력 / 주기적 체크포인트
  while (time < t_end && step < n_steps) {
    auto t00 = hrc::now();

    // [1] 적응 시간간격: 현재 속도장의 CFL 조건으로 dt = min(MaxCFL/속도, dt_cap)
    const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);
    // [2] NOB 변물성 갱신: 현재 온도 Tⁿ으로 μ, 1/ρ, κ, 1/(ρcp) 등을 다항식(글리세롤 모델)으로 재계산
    //     — 아래 에너지 방정식이 이 Tⁿ 물성을 사용
    physics::update_properties_gpu(gpu_props, gpu_fields[core::Var::T], pmodel);

    // [3] 에너지(온도) 방정식 — 대류 + 변수-κ 확산, 준음해 델타-폼:
    //     solve 안에서 sweep마다 A 밴드와 RHS를 동시에 조립(k_build_A_rhs_nob, 융합 build_A_RHS;
    //     stage-1이 명시 잔차 R을 인라인 계산) 후 3-sweep ADI(x,y,z 삼중대각 TDMA)로 δT를 풀어 T += δT
    //     sync_field = MPI 할로 교환 + 경계조건 재적용 (Tⁿ⁺¹ 완성)
    cudaDeviceSynchronize();
    auto t_s0 = hrc::now();
    equation::solve_scalar_nob_gpu(domain, bc, gpu_fields, gpu_props.KPP, gpu_props.dKPP,
                                   gpu_props.invRhoCp, gpu_props.dinvRhoCp, gpu_scalar, dt);
    core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);
    cudaDeviceSynchronize();
    t_scalar += ms_since(t_s0);

    // [4] 물성 재갱신 — 방금 푼 Tⁿ⁺¹로 μ, 1/ρ를 다시 계산해 운동량/부력이 Tⁿ⁺¹ 물성을 쓰게 함
    // Re-evaluate properties from the just-solved T^{n+1} so momentum viscosity/density
    // (mu, inv_rho) and the NOB buoyancy inv_rho use T^{n+1} — matches PaScaL_TCS
    // momentum_coeffi(T) called AFTER thermal_solver (MPM-STD staggered-time scheme).
    // Energy above already used the T^n properties from line 408.
    physics::update_properties_gpu(gpu_props, gpu_fields[core::Var::T], pmodel);

    // [5] 운동량 예측자(predictor) — 중간 속도 U*,V*,W* 계산:
    //     assemble  : 대류 + 변수-μ 확산의 명시 RHS 조립
    //     buoyancy  : 부력항 gz·(1/ρ̂)·T̂ 추가 (T는 W 위치로 면-보간; Report/34 수정)
    //     pressure_gradient : 증분투영 예측자 항 −dt·(1/ρ̂)∇Pⁿ 추가 (누적압력 P_acc 사용)
    //     solve     : 3-sweep ADI(TDMA) + 블록 하삼각 속도 커플링으로 증분 δU,δV,δW 계산
    //     update    : U += δU 등으로 중간 속도 완성 (아직 발산 ≠ 0)
    auto t_m0 = hrc::now();
    equation::assemble_momentum_var_visc_gpu(domain, gpu_fields, gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
    physics::add_buoyancy_force_nob_gpu(domain, gpu_momentum, gpu_fields[core::Var::T], gpu_props.invRho, buoy, dt);
    equation::add_pressure_gradient_nob_gpu(domain, gpu_momentum, gpu_P_acc, gpu_props.invRho, dt);
    equation::solve_momentum_var_gpu(domain, bc, gpu_fields, gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
    equation::update_velocity_gpu(gpu_fields, gpu_momentum);
    core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);   // 할로 교환 + BC 재적용 (U,V,W)
    core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
    cudaDeviceSynchronize();
    t_mom += ms_since(t_m0);
    // 디버그: 운동량 직후 상단 슬랩 |U|,|W| 최대값 (환경변수 MPMSTD_MOMDBG≥1일 때만 출력, 평소 no-op)
    equation::debug_slabmax_gpu("U.postmom", gpu_fields[core::Var::U], domain, 'U');
    equation::debug_slabmax_gpu("W.postmom", gpu_fields[core::Var::W], domain, 'W');

    // [6] 벽면 W=0 강제 — MAC 격자에서 z-벽의 W 면(내부 인덱스 k=h)은 TDMA에 안 접혀서
    //     솔브 후 명시적으로 0으로 눌러줘야 함
    // zero_w_wall: GPU kernel — no D2H/H2D
    cudaDeviceSynchronize();
    auto t_zw0 = hrc::now();
    {
      const auto nt = sub.n_total();
      const int n1t = nt[0], n2t = nt[1], n3t = nt[2], h = kHaloWidth;
      const bool has_lower = (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
      const bool has_upper = (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
      const dim3 blk(16, 16);
      const dim3 grd((n1t+15)/16, (n2t+15)/16);
      k_zero_w_wall_gpu<<<grd, blk>>>(gpu_fields[core::Var::W].data(), n2t, n3t, h, n1t, has_lower, has_upper);
    }
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
    cudaDeviceSynchronize();
    t_zero_w += ms_since(t_zw0);

    // [7] 압력 투영(incremental van Kan) — ∇·U*를 RHS로 Poisson을 풀어(δp)
    //     펜슬 FFT(x,y) + 분산 TDMA(z), 속도를 발산-제로로 투영, P_acc += δp
    // G4: GPU pressure solve (NOB — no H↔D transfers for pressure)
    // Report/25: faithful NOB pressure restored. ExtraTerm (Poisson) and the
    // (invRhoc-1)*∇dPhat velocity term are a cancelling PAIR (div=0 exact), not
    // double-counting; the 2026-07-02 OB swap was a misdiagnosis (θ_c unchanged).
    auto t_p0 = hrc::now();
    equation::solve_pressure_nob_gpu(domain, bc, gpu_fields, pressure, dt, gpu_props.invRho);
    equation::accumulate_pressure_gpu(gpu_P_acc, gpu_fields[core::Var::P]);  // Pⁿ⁺¹ = Pⁿ + δp
    cudaDeviceSynchronize();
    t_pressure += ms_since(t_p0);
    equation::debug_slabmax_gpu("U.postproj", gpu_fields[core::Var::U], domain);
    equation::debug_slabmax_gpu("W.postproj", gpu_fields[core::Var::W], domain);

    // [8] 온도 클리핑(NOB 전용, DeltaT>0일 때만) — 물성 다항식의 유효 범위를 벗어나는
    //     오버슛 T를 잘라냄; 몇 셀이 잘렸는지 d_clip_count에 기록해 모니터 때 보고
    // T clipping: GPU kernel — no D2H/H2D (interior only; ghosts hold the BC mirror)
    auto t_c0 = hrc::now();
    if (DeltaT > 0.0) {
      const auto nic = sub.n_interior();
      const auto ntc = sub.n_total();
      const dim3 blkc(8, 8, 8);
      const dim3 grdc((nic[0]+7)/8, (nic[1]+7)/8, (nic[2]+7)/8);
      if (!d_clip_count) {
        cudaMalloc(&d_clip_count, sizeof(unsigned long long));
        cudaMemset(d_clip_count, 0, sizeof(unsigned long long));
      }
      k_clip_T_gpu<<<grdc, blkc>>>(gpu_fields[core::Var::T].data(),
                                   nic[0], nic[1], nic[2], ntc[1], ntc[2],
                                   d_clip_count);
    }
    cudaDeviceSynchronize();
    t_clip += ms_since(t_c0);

    time += double(dt); ++step;   // 시간 전진 — 여기까지가 물리 스텝, 이후는 통계/출력

    // [9a] 통계 누적 (nstat_start 스텝 이후) — θ_c, Nu, z-프로파일 등을 디바이스 리덕션으로
    // GPU statistics: device reductions only — no field D2H (≤48 bytes per step)
    auto t_d0 = hrc::now();
    if (step >= nstat_start)
      gpu_accumulate_rbc(gpu_accum, stats, gpu_fields, topo.cart_comm());
    t_diag += ms_since(t_d0);

    // [9b] 모니터 출력 (nmonitor 스텝마다) — step/t/dt/div/θ_c/Nu 한 줄 + monitor.csv 기록
    auto t_a0 = hrc::now();
    if (step % nmonitor == 0) {
      gpu_print_step_rbc(gpu_accum, stats, gpu_fields, step, time, dt, alpha_T, mpi.is_root(), topo.cart_comm(), mon_fp);
      if (d_clip_count) {   // audit D3: report clip activations since last monitor
        unsigned long long local_clips = 0, total_clips = 0;
        cudaMemcpy(&local_clips, d_clip_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemset(d_clip_count, 0, sizeof(unsigned long long));
        MPI_Reduce(&local_clips, &total_clips, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, topo.cart_comm());
        if (mpi.is_root() && total_clips > 0)
          std::printf("  [clip] %llu cells clipped in the last %d steps\n", total_clips, nmonitor);
      }
    }
    t_accum += ms_since(t_a0);

    // [9c] 통계 파일 저장 (nstat_write 스텝마다) — stats_<step>.dat
    if (step >= nstat_start && nstat_write > 0 && step % nstat_write == 0) {
      const std::string sp = dir_stats + "stats_" + std::to_string(step) + ".dat";
      gpu_write_stats_rbc(gpu_accum, stats, sp, step, Ra, DeltaT,
                          mpi.is_root(), topo.cart_comm(), grid, sub);
    }

    // [9d] 주기적 재시작 체크포인트 — 필드를 D2H 복사 후 ckpt_a/ckpt_b 교대로 기록
    //      (중간에 잡이 죽어도 최소 한 벌은 온전; P는 누적압력 P_acc를 저장)
    // Periodic restart checkpoint into alternating ckpt_a/ckpt_b (see the
    // nrestart_write comment at config parse). Skips the final step — the
    // end-of-run write below covers it in <dir_out> itself.
    if (!restart_out.empty() && nrestart_write > 0
        && step % nrestart_write == 0 && step < n_steps && time < t_end) {
      gpu_fields[core::Var::U].to_host(fields[core::Var::U].data());
      gpu_fields[core::Var::V].to_host(fields[core::Var::V].data());
      gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
      gpu_fields[core::Var::T].to_host(fields[core::Var::T].data());
      gpu_P_acc.to_host(fields[core::Var::P].data());   // accumulated pressure Pⁿ
      const bool odd = ((step / nrestart_write) % 2) != 0;
      const std::string ckpt = restart_out + (odd ? "/ckpt_b" : "/ckpt_a");
      rbc::write_restart(fields, sub, stats, step, time, ckpt, topo.cart_comm());
    }

    // [10] 성능 프로파일 출력 — 첫 n_time_steps 스텝의 구간별(ms/step) 타이밍, 이후 정상상태 타이밍은 런 끝에
    if (step == n_time_steps && mpi.is_root()) {
      const double total = t_scalar+t_mom+t_zero_w+t_pressure+t_clip+t_diag+t_accum;
      std::printf("[TIMING] %d steps  total=%.1f ms/step\n", n_time_steps, total/n_time_steps);
      std::printf("  scalar=%.1f  momentum=%.1f  zero_w=%.1f  pressure=%.1f  clip=%.1f  diag_D2H=%.1f  accum=%.1f\n",
                  t_scalar/n_time_steps, t_mom/n_time_steps, t_zero_w/n_time_steps,
                  t_pressure/n_time_steps, t_clip/n_time_steps, t_diag/n_time_steps, t_accum/n_time_steps);
      // Reset accumulators to measure steady-state timing separately
      t_scalar=0; t_mom=0; t_zero_w=0; t_pressure=0; t_clip=0; t_diag=0; t_accum=0;
    }
    // Print steady-state timing snapshot (steps 11–n_steps) at end
    if (step == n_steps && mpi.is_root() && n_steps > n_time_steps) {
      const int ss_steps = n_steps - n_time_steps;
      const double total = t_scalar+t_mom+t_zero_w+t_pressure+t_clip+t_diag+t_accum;
      std::printf("[TIMING_SS] %d steps (steady-state)  total=%.1f ms/step\n", ss_steps, total/ss_steps);
      std::printf("  scalar=%.1f  momentum=%.1f  zero_w=%.1f  pressure=%.1f  clip=%.1f  diag_D2H=%.1f  accum=%.1f\n",
                  t_scalar/ss_steps, t_mom/ss_steps, t_zero_w/ss_steps,
                  t_pressure/ss_steps, t_clip/ss_steps, t_diag/ss_steps, t_accum/ss_steps);
    }
  }

  if (mon_fp) std::fclose(mon_fp);

  // Write final stats file
  {
    const std::string sp = dir_stats + "stats_final.dat";
    gpu_write_stats_rbc(gpu_accum, stats, sp, step, Ra, DeltaT,
                        mpi.is_root(), topo.cart_comm(), grid, sub);
  }

  // D2H z-profiles (once at run end, ≈0.5 KB each, for print_final)
  if (stats.tc_n > 0) {
    const int nzl = sub.n_interior()[2];
    cudaMemcpy(stats.t_sum_z.data()  + gpu_accum.koff, gpu_accum.d_t_sum_z,
               nzl*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stats.w2_sum_z.data() + gpu_accum.koff, gpu_accum.d_w2_sum_z,
               nzl*sizeof(double), cudaMemcpyDeviceToHost);
  }
  rbc::print_final(stats, sub, Ra, Pr, alpha_T, mpi.is_root(), "rbc_gpu_nob");

  // Write restart files (U/V/W/T/P + stats.txt) for continuation runs
  if (!restart_out.empty()) {
    gpu_fields[core::Var::U].to_host(fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_host(fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
    gpu_fields[core::Var::T].to_host(fields[core::Var::T].data());
    gpu_P_acc.to_host(fields[core::Var::P].data());   // accumulated pressure Pⁿ
    rbc::write_restart(fields, sub, stats, step, time, restart_out, topo.cart_comm());
  }
  return 0;
}
