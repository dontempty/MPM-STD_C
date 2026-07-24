// DHVC statistics infrastructure — device stat kernels + GpuDhvcAccum buffers +
// accum/monitor/write helpers, moved VERBATIM from apps/dhvc/main_gpu.cu.
// Numerics and file formats are unchanged; only wrapped in post::DhvcStatistics.
#include "post/statistics_dhvc.hpp"
#include "post/statistics.hpp"     // Stats, init/write/compute_re_delta_star

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace mpmstd::post {
namespace {

// ── GPU statistics kernels ────────────────────────────────────────────────────

__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* ull = (unsigned long long*)addr;
    unsigned long long old = *ull, assumed;
    do { assumed = old;
         old = atomicCAS(ull, assumed,
             __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

// Accumulate per-z-plane sums of U_cell, U²_cell, Wc_cell, P over xy.
// Grid: dim3(flat_blocks, nzl), 256 threads. Shared: 4×256 doubles.
__global__ void k_accum_z_dhvc(
    const double* U, const double* W, const double* P,
    double* d_u, double* d_u2, double* d_wc, double* d_p,
    int nx, int ny, int nz, int n2t, int n3t, int h, double wdt)
{
    extern __shared__ double shm[];
    const int kk  = blockIdx.y;
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid  = threadIdx.x, bsz = blockDim.x;
    double u_v=0, u2_v=0, wc_v=0, p_v=0;
    if (flat < nx*ny && kk < nz) {
        const int ii=flat/ny, jj=flat%ny, i=ii+h, j=jj+h, k=kk+h;
        // Cell-center interpolation (MAC grid)
        const double uc = 0.5*(U[(std::size_t)(i*n2t+j)*n3t+k]
                             + U[(std::size_t)((i+1)*n2t+j)*n3t+k]);
        const double wc = 0.5*(W[(std::size_t)(i*n2t+j)*n3t+k]
                             + W[(std::size_t)(i*n2t+j)*n3t+k+1]);
        u_v = uc; u2_v = uc*uc; wc_v = wc;
        p_v = P[(std::size_t)(i*n2t+j)*n3t+k];
    }
    shm[tid]          = u_v;
    shm[bsz+tid]      = u2_v;
    shm[2*bsz+tid]    = wc_v;
    shm[3*bsz+tid]    = p_v;
    __syncthreads();
    for (int s=bsz/2; s>0; s>>=1) {
        if (tid<s) {
            shm[tid]       += shm[tid+s];
            shm[bsz+tid]   += shm[bsz+tid+s];
            shm[2*bsz+tid] += shm[2*bsz+tid+s];
            shm[3*bsz+tid] += shm[3*bsz+tid+s];
        }
        __syncthreads();
    }
    if (tid==0 && kk<nz) {
        atomicAdd(&d_u[kk],  shm[0]*wdt);
        atomicAdd(&d_u2[kk], shm[bsz]*wdt);
        atomicAdd(&d_wc[kk], shm[2*bsz]*wdt);
        atomicAdd(&d_p[kk],  shm[3*bsz]*wdt);
    }
}

// Compute {|U|max, |V|max, |W|max, div_max} — d_out[4], must be zeroed before call.
__global__ void k_diag_dhvc(
    const double* U, const double* V, const double* W,
    const double* dx1, const double* dx2, const double* dx3,
    double* d_out, int nx, int ny, int nz, int n2t, int n3t, int h)
{
    const int ii=blockIdx.x*blockDim.x+threadIdx.x;
    const int jj=blockIdx.y*blockDim.y+threadIdx.y;
    const int kk=blockIdx.z*blockDim.z+threadIdx.z;
    if (ii>=nx||jj>=ny||kk>=nz) return;
    const int i=ii+h, j=jj+h, k=kk+h;
    const std::size_t c = (std::size_t)(i*n2t+j)*n3t+k;
    atomicMaxDouble(d_out+0, fabs(U[c]));
    atomicMaxDouble(d_out+1, fabs(V[c]));
    atomicMaxDouble(d_out+2, fabs(W[c]));
    const double dv =
        (U[(std::size_t)((i+1)*n2t+j)*n3t+k] - U[c]) / dx1[i] +
        (V[(std::size_t)(i*n2t+j+1)*n3t+k]   - V[c]) / dx2[j] +
        (W[(std::size_t)(i*n2t+j)*n3t+k+1]   - W[c]) / dx3[k];
    atomicMaxDouble(d_out+3, fabs(dv));
}

// Instantaneous global volume sums over interior cells (cell-centered, MAC interp):
//   d_g[0] = Σ ½(uc²+vc²+wc²)   (kinetic energy)
//   d_g[1] = Σ wc·θ             (wall-normal z convective heat flux → Nusselt)
//   d_g[2] = Σ θ²               (temperature variance → θ_rms)
// d_g[3] must be zeroed before call. Grid-stride flat loop → shared tree → atomicAdd.
__global__ void k_global_dhvc(
    const double* U, const double* V, const double* W, const double* T,
    double* d_g, int nx, int ny, int nz, int n2t, int n3t, int h)
{
    extern __shared__ double s[];
    const int tid = threadIdx.x, bsz = blockDim.x;
    const int ntot = nx*ny*nz;
    double ke=0, wt=0, tt=0;
    for (int idx = blockIdx.x*blockDim.x + tid; idx < ntot; idx += gridDim.x*bsz) {
        const int ii=idx/(ny*nz), jj=(idx/nz)%ny, kk=idx%nz;
        const int i=ii+h, j=jj+h, k=kk+h;
        const std::size_t c=(std::size_t)(i*n2t+j)*n3t+k;
        const double uc=0.5*(U[c]+U[(std::size_t)((i+1)*n2t+j)*n3t+k]);
        const double vc=0.5*(V[c]+V[(std::size_t)(i*n2t+j+1)*n3t+k]);
        const double wc=0.5*(W[c]+W[(std::size_t)(i*n2t+j)*n3t+k+1]);
        const double th=T[c];
        ke += 0.5*(uc*uc+vc*vc+wc*wc); wt += wc*th; tt += th*th;
    }
    s[tid]=ke; s[bsz+tid]=wt; s[2*bsz+tid]=tt; __syncthreads();
    for (int p=bsz/2; p>0; p>>=1) {
        if (tid<p) { s[tid]+=s[tid+p]; s[bsz+tid]+=s[bsz+tid+p]; s[2*bsz+tid]+=s[2*bsz+tid+p]; }
        __syncthreads();
    }
    if (tid==0) { atomicAdd(&d_g[0],s[0]); atomicAdd(&d_g[1],s[bsz]); atomicAdd(&d_g[2],s[2*bsz]); }
}

// ── GpuDhvcAccum: device stat buffers + GPU-side accumulation ────────────────
struct GpuDhvcAccum {
    double* d_u_sum  = nullptr;   // [nzl]  ΣΣ U_cell  (accumulated, never reset)
    double* d_u2_sum = nullptr;   // [nzl]  ΣΣ U²_cell
    double* d_wc_sum = nullptr;   // [nzl]  ΣΣ Wc_cell
    double* d_p_sum  = nullptr;   // [nzl]  ΣΣ P_cell
    double* d_diag   = nullptr;   // [4]    scratch: {umax,vmax,wmax,div_max}
    double* d_iu     = nullptr;   // [nzl]  scratch: instantaneous z-profile (U,U²,Wc,P)
    double* d_iu2    = nullptr;
    double* d_iwc    = nullptr;
    double* d_ip     = nullptr;
    double* d_glob   = nullptr;   // [3]    scratch: {KE_sum, Wθ_sum, θ²_sum}
    long    n_accum  = 0;
    double  t_accum  = 0.0;   // Σdt — accumulated physical time (time-average divisor)
    int nzl, nz_global, koff, nx, ny, nxg, nyg, n2t, n3t, h;
    std::vector<double> zc_local, zc_global;
    const double* d_dx1, *d_dx2, *d_dx3;

    GpuDhvcAccum(const parallel::mpi::Subdomain& sub, const grid::Grid& grid,
                 const core::GpuGridArrays& gg)
    {
        const auto ni=sub.n_interior(); nx=ni[0]; ny=ni[1]; nzl=ni[2];
        const auto nt=sub.n_total(); n2t=nt[1]; n3t=nt[2]; h=kHaloWidth;
        const auto ng=sub.n_global(); nxg=ng[0]; nyg=ng[1]; nz_global=ng[2];
        koff = sub.global_offset()[2];
        cudaMalloc(&d_u_sum,  nzl*sizeof(double)); cudaMemset(d_u_sum,  0, nzl*sizeof(double));
        cudaMalloc(&d_u2_sum, nzl*sizeof(double)); cudaMemset(d_u2_sum, 0, nzl*sizeof(double));
        cudaMalloc(&d_wc_sum, nzl*sizeof(double)); cudaMemset(d_wc_sum, 0, nzl*sizeof(double));
        cudaMalloc(&d_p_sum,  nzl*sizeof(double)); cudaMemset(d_p_sum,  0, nzl*sizeof(double));
        cudaMalloc(&d_diag,   4*sizeof(double));
        cudaMalloc(&d_iu,  nzl*sizeof(double));
        cudaMalloc(&d_iu2, nzl*sizeof(double));
        cudaMalloc(&d_iwc, nzl*sizeof(double));
        cudaMalloc(&d_ip,  nzl*sizeof(double));
        cudaMalloc(&d_glob, 3*sizeof(double));
        d_dx1=gg.d_dx1(); d_dx2=gg.d_dx2(); d_dx3=gg.d_dx3();
        const auto& zc = grid.xc(Direction::Z);
        zc_local.resize(nzl);
        for (int kl=0; kl<nzl; ++kl) zc_local[kl] = static_cast<double>(zc[h+kl]);
    }
    ~GpuDhvcAccum() {
        cudaFree(d_u_sum); cudaFree(d_u2_sum);
        cudaFree(d_wc_sum); cudaFree(d_p_sum);
        cudaFree(d_diag);
        cudaFree(d_iu); cudaFree(d_iu2); cudaFree(d_iwc); cudaFree(d_ip);
        cudaFree(d_glob);
    }
};

// Add one sample to device z-profile sums (no D2H).
inline void gpu_accum_dhvc(GpuDhvcAccum& ga, const core::GpuFields& gf, double dt) {
    const int flat_b = (ga.nx*ga.ny + 255)/256;
    // dt-weighted sample → true TIME-average: Σ(field·dt) divided by Σdt (=t_accum).
    k_accum_z_dhvc<<<dim3(flat_b, ga.nzl), 256, 4*256*sizeof(double)>>>(
        gf[core::Var::U].data(), gf[core::Var::W].data(), gf[core::Var::P].data(),
        ga.d_u_sum, ga.d_u2_sum, ga.d_wc_sum, ga.d_p_sum,
        ga.nx, ga.ny, ga.nzl, ga.n2t, ga.n3t, ga.h, dt);
    ga.t_accum += dt; ++ga.n_accum;
}

// Re_dstar from a globally gathered U_mean(z) profile.
static double re_dstar_from_profile(
    const std::vector<double>& U, const std::vector<double>& zc, double nu)
{
    const int nz = (int)U.size();
    if (nz <= 0) return 0.0;
    int kmax=0; double u_max=U[0];
    for (int k=0; k<nz; ++k) if (U[k] > u_max) { u_max=U[k]; kmax=k; }
    if (u_max <= 0.0) return 0.0;
    double dstar=0.0, z_prev=0.0, f_prev=1.0;
    for (int k=0; k<=kmax; ++k) {
        const double f = 1.0 - U[k]/u_max;
        dstar += 0.5*(f_prev + f)*(zc[k] - z_prev);
        z_prev=zc[k]; f_prev=f;
    }
    return u_max * dstar / nu;
}

// Monitor: GPU diag kernel + D2H 32 B, D2H u_sum ~nzl*8 B, compute Re_dstar, write CSV.
// Total D2H per call: ~1.6 KB regardless of domain size.
inline void gpu_monitor_dhvc(GpuDhvcAccum& ga, const core::GpuFields& gf,
    int step, double time, real_t dt, double nu, double alpha, double Ra,
    bool is_root, MPI_Comm comm, FILE* mon_fp,
    const std::vector<double>& zc_global,
    double cum_wall_s, double recent_ms_step)
{
    cudaMemset(ga.d_diag, 0, 4*sizeof(double));
    const dim3 blk(8,8,4), grd((ga.nx+7)/8,(ga.ny+7)/8,(ga.nzl+3)/4);
    k_diag_dhvc<<<grd,blk>>>(gf[core::Var::U].data(), gf[core::Var::V].data(),
                              gf[core::Var::W].data(),
                              ga.d_dx1, ga.d_dx2, ga.d_dx3, ga.d_diag,
                              ga.nx, ga.ny, ga.nzl, ga.n2t, ga.n3t, ga.h);
    cudaDeviceSynchronize();
    double loc4[4]; cudaMemcpy(loc4, ga.d_diag, 4*sizeof(double), cudaMemcpyDeviceToHost);
    double glb4[4]; MPI_Allreduce(loc4, glb4, 4, MPI_DOUBLE, MPI_MAX, comm);

    double re_dstar = 0.0;
    if (ga.t_accum > 0.0) {
        const double inv = 1.0/(ga.t_accum*ga.nxg*ga.nyg);   // time-average: / Σdt
        std::vector<double> u_loc(ga.nzl), send(ga.nz_global, 0.0), u_glb(ga.nz_global, 0.0);
        cudaMemcpy(u_loc.data(), ga.d_u_sum, ga.nzl*sizeof(double), cudaMemcpyDeviceToHost);
        for (int kl=0; kl<ga.nzl; ++kl) send[ga.koff+kl] = u_loc[kl]*inv;
        MPI_Allreduce(send.data(), u_glb.data(), ga.nz_global, MPI_DOUBLE, MPI_SUM, comm);
        re_dstar = re_dstar_from_profile(u_glb, zc_global, nu);
    }
    // ── Instantaneous diagnostics (THIS step, not accumulated): Re_dstar_inst,
    //    volume-avg KE, Nusselt (wall-normal Wθ flux), theta_rms. Run on ALL ranks
    //    (kernels + Allreduce) so the collectives stay collective. ──
    cudaMemset(ga.d_iu, 0, ga.nzl*sizeof(double));
    cudaMemset(ga.d_iu2,0, ga.nzl*sizeof(double));
    cudaMemset(ga.d_iwc,0, ga.nzl*sizeof(double));
    cudaMemset(ga.d_ip, 0, ga.nzl*sizeof(double));
    { const int fb=(ga.nx*ga.ny+255)/256;
      k_accum_z_dhvc<<<dim3(fb,ga.nzl),256,4*256*sizeof(double)>>>(
        gf[core::Var::U].data(), gf[core::Var::W].data(), gf[core::Var::P].data(),
        ga.d_iu, ga.d_iu2, ga.d_iwc, ga.d_ip,
        ga.nx, ga.ny, ga.nzl, ga.n2t, ga.n3t, ga.h, 1.0); }
    cudaMemset(ga.d_glob, 0, 3*sizeof(double));
    { const int ntot=ga.nx*ga.ny*ga.nzl, gb=std::min(512,(ntot+255)/256);
      k_global_dhvc<<<gb,256,3*256*sizeof(double)>>>(
        gf[core::Var::U].data(), gf[core::Var::V].data(), gf[core::Var::W].data(),
        gf[core::Var::T].data(), ga.d_glob,
        ga.nx, ga.ny, ga.nzl, ga.n2t, ga.n3t, ga.h); }
    cudaDeviceSynchronize();
    double re_inst=0.0;
    { const double inv=1.0/(double(ga.nxg)*ga.nyg);   // single sample (n=1)
      std::vector<double> u_loc(ga.nzl), send(ga.nz_global,0.0), u_glb(ga.nz_global,0.0);
      cudaMemcpy(u_loc.data(), ga.d_iu, ga.nzl*sizeof(double), cudaMemcpyDeviceToHost);
      for (int kl=0; kl<ga.nzl; ++kl) send[ga.koff+kl]=u_loc[kl]*inv;
      MPI_Allreduce(send.data(), u_glb.data(), ga.nz_global, MPI_DOUBLE, MPI_SUM, comm);
      re_inst = re_dstar_from_profile(u_glb, zc_global, nu); }
    double g3[3]; cudaMemcpy(g3, ga.d_glob, 3*sizeof(double), cudaMemcpyDeviceToHost);
    double g3g[3]; MPI_Allreduce(g3, g3g, 3, MPI_DOUBLE, MPI_SUM, comm);
    const double ncell = double(ga.nxg)*ga.nyg*ga.nz_global;
    const double KE   = g3g[0]/ncell;
    const double Nu   = 1.0 + (g3g[1]/ncell)/alpha;   // wall-normal convective Nu
    const double Trms = std::sqrt(g3g[2]/ncell);

    if (!is_root) return;
    std::printf("step=%d t=%.3g dt=%.3g div=%.2e Re=%.3f(avg=%.3f) KE=%.3e Nu=%.3f Trms=%.3f wall=%.0fs ms/step=%.1f\n",
                step, time, double(dt), glb4[3], re_inst, re_dstar, KE, Nu, Trms, cum_wall_s, recent_ms_step);
    if (mon_fp) {
        // columns: step,time,dt,div,Re_avg,Re_inst,KE,Nu,Trms,wall_s,ms_per_step
        std::fprintf(mon_fp, "%d,%.6g,%.6g,%.6e,%.6f,%.6f,%.6e,%.6f,%.6f,%.3f,%.3f\n",
                     step, time, double(dt), glb4[3], re_dstar, re_inst, KE, Nu, Trms,
                     cum_wall_s, recent_ms_step);
        std::fflush(mon_fp);
    }
}

// Write stats .dat file: D2H all 4 sums (~6 KB), gather via MPI, write Tecplot.
inline void gpu_write_stats_dhvc(GpuDhvcAccum& ga, Stats& s,
    const std::string& path, int step, double nu, const core::Domain& domain)
{
    if (ga.t_accum <= 0.0) return;
    const double inv = 1.0/(ga.t_accum*ga.nxg*ga.nyg);   // time-average: / Σdt
    std::vector<double> u_loc(ga.nzl), u2_loc(ga.nzl), wc_loc(ga.nzl), p_loc(ga.nzl);
    cudaMemcpy(u_loc.data(),  ga.d_u_sum,  ga.nzl*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u2_loc.data(), ga.d_u2_sum, ga.nzl*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(wc_loc.data(), ga.d_wc_sum, ga.nzl*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_loc.data(),  ga.d_p_sum,  ga.nzl*sizeof(double), cudaMemcpyDeviceToHost);
    s.n = ga.n_accum;
    for (int kl=0; kl<ga.nzl; ++kl) {
        s.U_m[kl]   = u_loc[kl]  * inv;
        s.U2_m[kl]  = u2_loc[kl] * inv;
        s.Wc_m[kl]  = wc_loc[kl] * inv;
        s.P_m[kl]   = p_loc[kl]  * inv;
        s.V_m[kl]=0; s.V2_m[kl]=0; s.Wc2_m[kl]=0; s.UWc_m[kl]=0;
    }
    write_statistics_cpu(s, path, step, nu, domain);
}
} // anonymous namespace

// ── DhvcStatistics: PImpl wrapper over the moved infrastructure ──────────────
struct DhvcStatistics::Impl {
  DhvcStatisticsConfig cfg;
  MPI_Comm comm;
  bool is_root;
  const core::Domain& domain;
  GpuDhvcAccum accum;
  Stats stats;
  std::FILE* mon_fp = nullptr;
  double t_next_out;

  Impl(const parallel::mpi::Subdomain& sub, const grid::Grid& grid,
       const core::GpuGridArrays& gg, const core::Domain& dom,
       const DhvcStatisticsConfig& c, MPI_Comm cm, bool root)
    : cfg(c), comm(cm), is_root(root), domain(dom),
      accum(sub, grid, gg), t_next_out(c.t_stat_start) {
    init_statistics_cpu(stats, domain);
    if (is_root) {
      mon_fp = std::fopen((cfg.dir_stats + "monitor.csv").c_str(), "w");
      if (mon_fp) std::fprintf(mon_fp, "step,t,dt,div,Re_dstar,Re_inst,KE,Nu,theta_rms\n");
    }
  }
  ~Impl() { if (mon_fp) std::fclose(mon_fp); }
};

DhvcStatistics::DhvcStatistics(const parallel::mpi::Subdomain& sub, const grid::Grid& grid,
                               const core::GpuGridArrays& gg, const core::Domain& domain,
                               const DhvcStatisticsConfig& cfg, MPI_Comm comm, bool is_root)
  : impl_(std::make_unique<Impl>(sub, grid, gg, domain, cfg, comm, is_root)) {}

DhvcStatistics::~DhvcStatistics() = default;

void DhvcStatistics::accumulate_if_due(double time, real_t dt, const core::GpuFields& gf) {
  if (time >= impl_->cfg.t_stat_start)
    gpu_accum_dhvc(impl_->accum, gf, static_cast<double>(dt));
}

void DhvcStatistics::monitor_if_due(int step, double time, real_t dt, const core::GpuFields& gf,
                                    double cum_wall_s, double recent_ms_step) {
  if (step % impl_->cfg.nmonitor != 0) return;
  gpu_monitor_dhvc(impl_->accum, gf, step, time, dt, impl_->cfg.nu, impl_->cfg.alpha_T,
                   impl_->cfg.Ra, impl_->is_root, impl_->comm, impl_->mon_fp,
                   impl_->stats.zc_global, cum_wall_s, recent_ms_step);
}

void DhvcStatistics::write_profile_if_due(double time, int step) {
  if (time >= impl_->t_next_out && impl_->accum.t_accum > 0.0) {
    gpu_write_stats_dhvc(impl_->accum, impl_->stats,
        impl_->cfg.dir_stats + "stats_t" + std::to_string((long)std::llround(time)) + ".dat",
        step, impl_->cfg.nu, impl_->domain);
    impl_->t_next_out += impl_->cfg.t_stat_out;
  }
}

double DhvcStatistics::final_re_delta_star() const {
  return compute_re_delta_star_cpu(impl_->stats, impl_->cfg.nu, impl_->domain);
}

} // namespace mpmstd::post
