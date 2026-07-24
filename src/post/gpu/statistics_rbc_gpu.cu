// RBC statistics infrastructure — device stat kernels + GpuRbcAccum buffers +
// accumulate/monitor/write helpers, moved VERBATIM from apps/rbc/main_gpu_nob.cu
// (F4). Numerics and file formats are unchanged; only wrapped in
// post::RbcStatistics. rbc::RbcStats became the local RbcRunning so post/ keeps
// no dependency on the app layer.
#include "post/statistics_rbc.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace mpmstd::post {
namespace {

// Running sums (shape-identical to the app's rbc::RbcStats, kept local here).
struct RbcRunning {
  double tc_sum = 0.0; long tc_n = 0;
  std::vector<double> t_sum_z, w2_sum_z;
  double wt_sum = 0.0;
  explicit RbcRunning(int nzg) : t_sum_z(nzg, 0.0), w2_sum_z(nzg, 0.0) {}
};

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

// Accumulate z-profiles (per z-plane sums over i,j): ∑T,∑T², ∑U,∑U², ∑V,∑V², ∑W,∑W².
// Later → T_mean, T_rms, U/V/W_mean, U/V/W_rms(=√(⟨·²⟩−⟨·⟩²)).
// Grid: (ceil(nx*ny/256), nz) blocks, 256 threads, 8*256*sizeof(double) shared mem.
// 주: U/V/W는 면-엇갈림 값을 그대로 씀 (기존 W_rms 관례와 동일; 셀중심 보간 안 함).
__global__ void k_accum_z_profile_rbc(
    const double* T, const double* U, const double* V, const double* W,
    double* d_t_sum_z, double* d_t2_sum_z,
    double* d_u_sum_z, double* d_u2_sum_z,
    double* d_v_sum_z, double* d_v2_sum_z,
    double* d_w_sum_z, double* d_w2_sum_z,
    int nx, int ny, int nz, int n2t, int n3t, int h)
{
  extern __shared__ double shm[];
  const int kk = blockIdx.y;
  const int flat = blockIdx.x*blockDim.x+threadIdx.x;
  const int tid = threadIdx.x, bsz = blockDim.x;
  double tv=0.0, t2v=0.0, uv=0.0, u2v=0.0, vv=0.0, v2v=0.0, wv=0.0, w2v=0.0;
  if (flat<nx*ny && kk<nz) {
    const int ii=flat/ny, jj=flat%ny, i=ii+h, j=jj+h, k=kk+h;
    const std::size_t c = (std::size_t)(i*n2t+j)*n3t+k;
    const double tval=T[c]; tv=tval; t2v=tval*tval;
    const double uval=U[c]; uv=uval; u2v=uval*uval;
    const double vval=V[c]; vv=vval; v2v=vval*vval;
    const double wval=W[c]; wv=wval; w2v=wval*wval;
  }
  shm[tid]=tv;         shm[bsz+tid]=t2v;
  shm[2*bsz+tid]=uv;   shm[3*bsz+tid]=u2v;
  shm[4*bsz+tid]=vv;   shm[5*bsz+tid]=v2v;
  shm[6*bsz+tid]=wv;   shm[7*bsz+tid]=w2v;
  __syncthreads();
  for (int s=bsz/2; s>0; s>>=1) {
    if (tid<s) {
      for (int lane=0; lane<8; ++lane) shm[lane*bsz+tid] += shm[lane*bsz+tid+s];
    }
    __syncthreads();
  }
  if (tid==0 && kk<nz) {
    atomicAdd(&d_t_sum_z[kk],  shm[0]);       atomicAdd(&d_t2_sum_z[kk], shm[bsz]);
    atomicAdd(&d_u_sum_z[kk],  shm[2*bsz]);   atomicAdd(&d_u2_sum_z[kk], shm[3*bsz]);
    atomicAdd(&d_v_sum_z[kk],  shm[4*bsz]);   atomicAdd(&d_v2_sum_z[kk], shm[5*bsz]);
    atomicAdd(&d_w_sum_z[kk],  shm[6*bsz]);   atomicAdd(&d_w2_sum_z[kk], shm[7*bsz]);
  }
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
  double* d_t_sum_z = nullptr;   // [nzl] ∑T  → T_mean
  double* d_t2_sum_z= nullptr;   // [nzl] ∑T² → T_rms
  double* d_u_sum_z = nullptr;   // [nzl] ∑U  → U_mean
  double* d_u2_sum_z= nullptr;   // [nzl] ∑U² → U_rms
  double* d_v_sum_z = nullptr;   // [nzl] ∑V  → V_mean
  double* d_v2_sum_z= nullptr;   // [nzl] ∑V² → V_rms
  double* d_w_sum_z = nullptr;   // [nzl] ∑W  → W_mean
  double* d_w2_sum_z= nullptr;   // [nzl] ∑W² → W_rms
  int nzl, nx, ny, nz, n2t, n3t, h, kc_local, nxg, nyg, koff, nz_global;
  const double* d_dx1; const double* d_dx2; const double* d_dx3;

  GpuRbcAccum(const parallel::mpi::Subdomain& sub, const core::GpuGridArrays& gg) {
    const auto ni = sub.n_interior(); nx=ni[0]; ny=ni[1]; nz=ni[2]; nzl=nz;
    const auto nt = sub.n_total();    n2t=nt[1]; n3t=nt[2]; h=kHaloWidth;
    const auto ng = sub.n_global();   nxg=ng[0]; nyg=ng[1]; nz_global=ng[2];
    const int nzg = ng[2], kc_g = nzg/2;
    koff = sub.global_offset()[2];
    kc_local = (kc_g>=koff && kc_g<koff+nzl) ? (kc_g - koff) : -1;
    cudaMalloc(&d_tmp2,     2*sizeof(double));
    cudaMalloc(&d_diag6,    6*sizeof(double));
    for (double** p : {&d_t_sum_z,&d_t2_sum_z,&d_u_sum_z,&d_u2_sum_z,
                       &d_v_sum_z,&d_v2_sum_z,&d_w_sum_z,&d_w2_sum_z}) {
      cudaMalloc(p, nzl*sizeof(double));
      cudaMemset(*p, 0, nzl*sizeof(double));
    }
    d_dx1=gg.d_dx1(); d_dx2=gg.d_dx2(); d_dx3=gg.d_dx3();
  }
  ~GpuRbcAccum() {
    cudaFree(d_tmp2); cudaFree(d_diag6);
    cudaFree(d_t_sum_z); cudaFree(d_t2_sum_z);
    cudaFree(d_u_sum_z); cudaFree(d_u2_sum_z);
    cudaFree(d_v_sum_z); cudaFree(d_v2_sum_z);
    cudaFree(d_w_sum_z); cudaFree(d_w2_sum_z);
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
inline void gpu_accumulate_rbc(GpuRbcAccum& ga, RbcRunning& s,
    const core::GpuFields& gf, MPI_Comm comm)
{
  const double* T = gf[core::Var::T].data();
  const double* U = gf[core::Var::U].data();
  const double* V = gf[core::Var::V].data();
  const double* W = gf[core::Var::W].data();
  double wt_instant, tc_instant;
  gpu_compute_instant_tc_wt(ga, gf, comm, wt_instant, tc_instant);
  // k_accum_z_profile: 256 threads, grid (ceil(nx*ny/256), nz), 8 reduction lanes
  const int flat_blocks = (ga.nx*ga.ny+255)/256;
  k_accum_z_profile_rbc<<<dim3(flat_blocks,ga.nz),256,8*256*sizeof(double)>>>(
      T, U, V, W,
      ga.d_t_sum_z, ga.d_t2_sum_z, ga.d_u_sum_z, ga.d_u2_sum_z,
      ga.d_v_sum_z, ga.d_v2_sum_z, ga.d_w_sum_z, ga.d_w2_sum_z,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h);
  cudaDeviceSynchronize();
  s.wt_sum += wt_instant;
  s.tc_sum += tc_instant;
  ++s.tc_n;
}

// GPU print_step: flow_diag + div_max on device, D2H 48 bytes, MPI_Allreduce → printf + CSV.
// Total D2H per call: 48 bytes.
inline void gpu_print_step_rbc(GpuRbcAccum& ga, const RbcRunning& s,
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
  // regardless of the stats window, so it's meaningful pre-stats-window too.
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
inline void gpu_write_stats_rbc(GpuRbcAccum& ga, const RbcRunning& s,
    const std::string& path, int step, double Ra, double DeltaT,
    bool is_root, MPI_Comm comm,
    const grid::Grid& grid, const parallel::mpi::Subdomain& sub)
{
  if (s.tc_n == 0) return;
  const int nzl = ga.nzl, nzg = ga.nz_global;
  const double inv = 1.0 / (s.tc_n * static_cast<double>(ga.nxg) * ga.nyg);
  // D2H local z-profiles (∑T,∑T², ∑U,∑U², ∑V,∑V², ∑W,∑W²)
  auto d2h = [&](const double* dev){ std::vector<double> h(nzl);
    cudaMemcpy(h.data(), dev, nzl*sizeof(double), cudaMemcpyDeviceToHost); return h; };
  const std::vector<double> t_loc=d2h(ga.d_t_sum_z),  t2_loc=d2h(ga.d_t2_sum_z),
                            u_loc=d2h(ga.d_u_sum_z),  u2_loc=d2h(ga.d_u2_sum_z),
                            v_loc=d2h(ga.d_v_sum_z),  v2_loc=d2h(ga.d_v2_sum_z),
                            w_loc=d2h(ga.d_w_sum_z),  w2_loc=d2h(ga.d_w2_sum_z);
  // Gather global profiles (each *inv = plane+time mean of the summed quantity)
  auto gather = [&](const std::vector<double>& loc){
    std::vector<double> send(nzg,0.0), glb(nzg,0.0);
    for (int kl=0; kl<nzl; ++kl) send[ga.koff+kl] = loc[kl]*inv;
    MPI_Allreduce(send.data(), glb.data(), nzg, MPI_DOUBLE, MPI_SUM, comm);
    return glb;
  };
  const std::vector<double> t_glb=gather(t_loc),  t2_glb=gather(t2_loc),
                            u_glb=gather(u_loc),  u2_glb=gather(u2_loc),
                            v_glb=gather(v_loc),  v2_glb=gather(v2_loc),
                            w_glb=gather(w_loc),  w2_glb=gather(w2_loc);
  if (!is_root) return;
  FILE* fp = std::fopen(path.c_str(), "w");
  if (!fp) { std::fprintf(stderr, "[rbc_stats] cannot open '%s'\n", path.c_str()); return; }
  std::fprintf(fp, "# RBC z-profile stats (step=%d, n=%ld, Ra=%.2e, DeltaT=%.1f)\n",
               step, s.tc_n, Ra, DeltaT);
  std::fprintf(fp, "# Z,T_mean,T_rms,U_mean,U_rms,V_mean,V_rms,W_mean,W_rms\n");
  auto rms = [](double sq, double mean){ return std::sqrt(std::max(sq - mean*mean, 0.0)); };
  for (int k=0; k<nzg; ++k) {
    const double zc = static_cast<double>(k+1) / static_cast<double>(nzg);  // normalized z
    std::fprintf(fp, "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                 zc,
                 t_glb[k], rms(t2_glb[k], t_glb[k]),
                 u_glb[k], rms(u2_glb[k], u_glb[k]),
                 v_glb[k], rms(v2_glb[k], v_glb[k]),
                 w_glb[k], rms(w2_glb[k], w_glb[k]));
  }
  std::fclose(fp);
}

// print_final — end-of-run summary, moved verbatim from rbc::print_final
// (rbc_common.hpp) so post/ needs no app dependency.
inline void print_final_rbc(const RbcRunning& s, const parallel::mpi::Subdomain& sub,
                            double Ra, double Pr, double alpha_T, bool is_root, const char* tag) {
  const double theta_c_out = (s.tc_n > 0) ? s.tc_sum / s.tc_n : 0.0;
  if (is_root) std::printf("[%s] done: Ra=%.2e Pr=%.4g  Theta_c=%.5f\n", tag, Ra, Pr, theta_c_out);
  if (s.tc_n <= 0) return;
  const int nzg = static_cast<int>(s.t_sum_z.size());
  const int nxg = sub.n_global()[0], nyg = sub.n_global()[1];
  std::vector<double> t_z_g(nzg, 0.0), w2_z_g(nzg, 0.0);
  MPI_Allreduce(s.t_sum_z.data(),  t_z_g.data(),  nzg, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  MPI_Allreduce(s.w2_sum_z.data(), w2_z_g.data(), nzg, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  const double inv_nxy = 1.0 / (double(nxg)*nyg), inv_n = 1.0 / s.tc_n;
  const double Nu = 1.0 + s.wt_sum * inv_nxy * inv_n / alpha_T;
  if (!is_root) return;
  std::printf("[%s] Nu=%.4f  (wt_mean=%.6f  alpha_T=%.4e  tc_n=%ld)\n",
              tag, Nu, s.wt_sum * inv_nxy * inv_n, alpha_T, s.tc_n);
  std::printf("[%s] k  T_mean  W_rms\n", tag);
  for (int k = 0; k < nzg; ++k)
    std::printf("  %3d  %+.5f  %.5f\n", k, t_z_g[k]*inv_nxy*inv_n,
                std::sqrt(std::max(0.0, w2_z_g[k]*inv_nxy*inv_n)));
}

} // anonymous namespace

// ── RbcStatistics: PImpl wrapper over the moved infrastructure ───────────────
struct RbcStatistics::Impl {
  RbcStatisticsConfig cfg;
  MPI_Comm comm;
  bool is_root;
  const parallel::mpi::Subdomain& sub;
  const grid::Grid& grid;
  GpuRbcAccum accum;
  RbcRunning  run;
  std::FILE* mon_fp = nullptr;
  double t_next_out;
  unsigned long long* d_clip_count = nullptr;

  Impl(const parallel::mpi::Subdomain& s, const core::GpuGridArrays& gg,
       const grid::Grid& g, const RbcStatisticsConfig& c, MPI_Comm cm, bool root)
    : cfg(c), comm(cm), is_root(root), sub(s), grid(g),
      accum(s, gg), run(s.n_global()[2]), t_next_out(c.t_stat_start) {
    if (is_root) {
      const std::string mon_path = cfg.dir_stats + "monitor.csv";
      mon_fp = std::fopen(mon_path.c_str(), "w");
      if (mon_fp) std::fprintf(mon_fp, "step,t,dt,div,Theta_c,Nu\n");
    }
  }
  ~Impl() {
    if (mon_fp) std::fclose(mon_fp);
    if (d_clip_count) cudaFree(d_clip_count);   // accum frees its own device buffers
  }
};

RbcStatistics::RbcStatistics(const parallel::mpi::Subdomain& sub,
                             const core::GpuGridArrays& gg, const grid::Grid& grid,
                             const RbcStatisticsConfig& cfg, MPI_Comm comm, bool is_root)
  : impl_(std::make_unique<Impl>(sub, gg, grid, cfg, comm, is_root)) {}

RbcStatistics::~RbcStatistics() = default;

void RbcStatistics::accumulate_if_due(double time, const core::GpuFields& gf) {
  if (time >= impl_->cfg.t_stat_start)
    gpu_accumulate_rbc(impl_->accum, impl_->run, gf, impl_->comm);
}

void RbcStatistics::monitor_if_due(int step, double time, real_t dt,
                                   const core::GpuFields& gf) {
  if (step % impl_->cfg.nmonitor != 0) return;
  gpu_print_step_rbc(impl_->accum, impl_->run, gf, step, time, dt,
                     impl_->cfg.alpha_T, impl_->is_root, impl_->comm, impl_->mon_fp);
  if (impl_->d_clip_count) {   // audit D3: report clip activations since last monitor
    unsigned long long local_clips = 0, total_clips = 0;
    cudaMemcpy(&local_clips, impl_->d_clip_count, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaMemset(impl_->d_clip_count, 0, sizeof(unsigned long long));
    MPI_Reduce(&local_clips, &total_clips, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               impl_->comm);
    if (impl_->is_root && total_clips > 0)
      std::printf("  [clip] %llu cells clipped in the last %d steps\n",
                  total_clips, impl_->cfg.nmonitor);
  }
}

void RbcStatistics::write_profile_if_due(double time, int step) {
  if (time >= impl_->t_next_out && impl_->run.tc_n > 0) {
    const std::string sp = impl_->cfg.dir_stats + "stats_t"
                         + std::to_string((long)std::llround(time)) + ".dat";
    gpu_write_stats_rbc(impl_->accum, impl_->run, sp, step, impl_->cfg.Ra,
                        impl_->cfg.DeltaT, impl_->is_root, impl_->comm,
                        impl_->grid, impl_->sub);
    impl_->t_next_out += impl_->cfg.t_stat_out;
  }
}

void RbcStatistics::finalize(int step) {
  gpu_write_stats_rbc(impl_->accum, impl_->run, impl_->cfg.dir_stats + "stats_final.dat",
                      step, impl_->cfg.Ra, impl_->cfg.DeltaT, impl_->is_root,
                      impl_->comm, impl_->grid, impl_->sub);
  // D2H z-profiles (once at run end, ~0.5 KB each, for print_final)
  if (impl_->run.tc_n > 0) {
    const int nzl = impl_->sub.n_interior()[2];
    cudaMemcpy(impl_->run.t_sum_z.data()  + impl_->accum.koff, impl_->accum.d_t_sum_z,
               nzl*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(impl_->run.w2_sum_z.data() + impl_->accum.koff, impl_->accum.d_w2_sum_z,
               nzl*sizeof(double), cudaMemcpyDeviceToHost);
  }
  print_final_rbc(impl_->run, impl_->sub, impl_->cfg.Ra, impl_->cfg.Pr,
                  impl_->cfg.alpha_T, impl_->is_root, "rbc_gpu_nob");
}

std::vector<std::pair<std::string, double>> RbcStatistics::export_extras() const {
  return { {"tc_sum", impl_->run.tc_sum},
           {"tc_n",   static_cast<double>(impl_->run.tc_n)},
           {"wt_sum", impl_->run.wt_sum} };
}

void RbcStatistics::import_extras(const RestartState& state, bool stats_reset) {
  if (stats_reset) {
    if (impl_->is_root)
      std::printf("[restart] stats_reset=1 — starting Theta_c/Nu accumulation fresh from step=%d\n",
                  state.step);
    return;
  }
  for (const auto& e : state.extras) {
    if      (e.first == "tc_sum") impl_->run.tc_sum = e.second;
    else if (e.first == "tc_n")   impl_->run.tc_n   = static_cast<long>(e.second);
    else if (e.first == "wt_sum") impl_->run.wt_sum = e.second;
  }
}

unsigned long long* RbcStatistics::clip_counter() {
  if (!impl_->d_clip_count) {
    cudaMalloc(&impl_->d_clip_count, sizeof(unsigned long long));
    cudaMemset(impl_->d_clip_count, 0, sizeof(unsigned long long));
  }
  return impl_->d_clip_count;
}

} // namespace mpmstd::post
