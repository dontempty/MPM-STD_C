// apps/dhvc — GPU backend, OB differentially-heated vertical convection (PaScaL_TCS Fig 7).
//
// Axis map (identical to CPU main.cpp):
//   x = streamwise / vertical / gravity (periodic)  ← buoyancy on U  (gravity_axis=0)
//   y = spanwise                        (periodic)
//   z = wall-normal (hot z=0, cold z=Lz)            (wall)
//
// GPU time loop: scalar → momentum+buoyancy → mass-flow correction → pressure.
// Statistics accumulated via GPU kernels (no full-field D2H during time loop).
// Monitor CSV + stats .dat written from device-reduced data.

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/boundary_ops.hpp"
#include "core/variables.hpp"
#include "equation/momentum/linear_system.hpp"
#include "equation/scalar/linear_system.hpp"
#include "equation/pressure/linear_system.hpp"
#include "core/config.hpp"
#include "core/field_gpu.hpp"
#include "core/gpu/gpu_grid.hpp"
#include "core/gpu_binding.hpp"

#include "config/logger.hpp"
#include "boundary/problem_loader.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"

#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/pressure/solve.hpp"
#include "physics/buoyancy/buoyancy.hpp"
#include "physics/forcing/forcing.hpp"
#include "driver/cfl.hpp"
#include "physics/properties/properties.hpp"
#include "post/statistics.hpp"
#include "post/io.hpp"            // write_restart_cpu (checkpoint)

#include <cuda_runtime.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mpmstd;

namespace {

// ── Wall-normal W zeroing kernel ──────────────────────────────────────────────
__global__ void k_zero_w_wall(real_t* W, int n1, int n2, int n3, int h,
                               bool has_lo, bool has_hi) {
    const int ni = n1 - 2*h, nj = n2 - 2*h;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= ni * nj) return;
    const int i = idx / nj + h, j = idx % nj + h;
    if (has_lo) W[(i * n2 + j) * n3 + h]        = real_t(0);
    if (has_hi) W[(i * n2 + j) * n3 + (n3 - h)] = real_t(0);
}

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
inline void gpu_write_stats_dhvc(GpuDhvcAccum& ga, post::Stats& s,
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
    post::write_statistics_cpu(s, path, step, nu, domain);
}

} // anonymous namespace

int main(int argc, char** argv) {
    parallel::mpi::MpiContext mpi(&argc, &argv);
    config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
    const std::string input_path = (argc >= 2) ? argv[1] : "apps/dhvc/input.toml";
    auto cfg = config::Config::load(input_path);

    auto is_per = [](const std::string& s){ return s == "periodic"; };
    std::array<int,3>  dims     = {cfg.get<int>("mpi","np1"), cfg.get<int>("mpi","np2"), cfg.get<int>("mpi","np3")};
    std::array<bool,3> periodic = {is_per(cfg.get_or<std::string>("topology","x","periodic")),
                                   is_per(cfg.get_or<std::string>("topology","y","periodic")),
                                   is_per(cfg.get_or<std::string>("topology","z","wall"))};
    parallel::mpi::MpiTopology topo(mpi, dims, periodic);
    std::array<int,3> n_global = {cfg.get<int>("mesh","n1m"), cfg.get<int>("mesh","n2m"), cfg.get<int>("mesh","n3m")};
    parallel::mpi::Subdomain sub(topo, n_global);

    std::array<grid::AxisConfig,3> axes;
    for (int a = 0; a < 3; ++a) {
        const int    is_uni = cfg.get_or<int>   ("uniform_mesh","uniform"+std::to_string(a+1), 1);
        const double gamma  = cfg.get_or<double>("mesh_stretch","gamma" +std::to_string(a+1), 0.0);
        const double L      = cfg.get<double>   ("domain", std::string("L")+char('x'+a));
        axes[a] = {n_global[a], L, is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma};
    }
    grid::Grid grid(sub, axes);
    boundary::BoundaryCondition bc = boundary::load_problem_from_config(cfg);
    bc.validate();
    auto  tdma_owner = linear_solver::tdma::TdmaRegistry::make_default(topo);
    core::Domain domain{grid, topo, sub, *tdma_owner};

    const double Ra      = cfg.get<double>("physics","Ra");
    const double Pr      = cfg.get<double>("physics","Pr");
    const double nu      = std::sqrt(Pr / Ra);
    const double alpha_T = 1.0 / std::sqrt(Ra * Pr);

    const double t_end      = cfg.get_or<double>("time","t_end",       400.0);
    const real_t dt_cap     = static_cast<real_t>(cfg.get_or<double>("time","dt_start",  0.02));
    const real_t max_cfl    = static_cast<real_t>(cfg.get_or<double>("time","MaxCFL",    0.5));
    const int    n_steps    = cfg.get_or<int>   ("time","n_steps",    200000);
    const int    nmonitor   = cfg.get_or<int>("output","nmonitor",    500);
    // Statistics keyed on PHYSICAL TIME t (not step) → independent of the variable dt.
    // Accumulate from t>=t_stat_start; write a cumulative-average stats file every
    // t_stat_out of physical time. Accumulation is dt-weighted (true time-average,
    // sampled EVERY step) so it is fully independent of the variable dt.
    const double t_stat_start = cfg.get_or<double>("output","t_stat_start", 30.0);
    const double t_stat_out   = cfg.get_or<double>("output","t_stat_out", 30.0);
    double       t_next_out   = t_stat_start;   // next physical time to write stats
    const std::string dir_stats = cfg.get_or<std::string>("output","dir_stats","apps/dhvc/statistics/");
    const std::string restart_out = cfg.get_or<std::string>("restart","dir_out","");  // checkpoint at run end
    const std::string restart_in  = cfg.get_or<std::string>("restart","dir_in","");   // warm-start source (any mesh)
    if (mpi.is_root()) { std::error_code ec; std::filesystem::create_directories(dir_stats, ec); }
    MPI_Barrier(topo.cart_comm());

    // CPU fields: used only for IC setup + stats D2H populate
    core::CpuFields cpu_fields;
    cpu_fields.add(core::Var::U, sub); cpu_fields.add(core::Var::V, sub);
    cpu_fields.add(core::Var::W, sub); cpu_fields.add(core::Var::P, sub);
    cpu_fields.add(core::Var::T, sub);
    cpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
    cpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

    // Initial condition: analytical base state + perturbation (cold start), OR a
    // warm start interpolated from a previous (possibly coarser) run's restart.
    if (restart_in.empty()) {
        const double Lx=cfg.get<double>("domain","Lx"), Ly=cfg.get<double>("domain","Ly"), Lz=cfg.get<double>("domain","Lz");
        const double T_lo=bc.T.face(Direction::Z,Side::Minus).value;
        const double T_hi=bc.T.face(Direction::Z,Side::Plus ).value;
        const double amp=cfg.get_or<double>("init","perturb",0.05);
        const auto& xc=grid.xc(Direction::X); const auto& yc=grid.xc(Direction::Y); const auto& zc=grid.xc(Direction::Z);
        const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2];
        real_t* U=cpu_fields[core::Var::U].data(), *V=cpu_fields[core::Var::V].data();
        real_t* W=cpu_fields[core::Var::W].data(), *T=cpu_fields[core::Var::T].data();
        for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) for (int k=0;k<n3;++k) {
            const int c=(i*n2+j)*n3+k;
            const double z=zc[k], x=xc[i], y=yc[j];
            U[c]=static_cast<real_t>(z*(2.0*z/Lz-1.0)*(z/Lz-1.0)/(12.0*nu));
            T[c]=static_cast<real_t>(T_lo+(T_hi-T_lo)*(z/Lz));
            const double env=std::sin(M_PI*z/Lz);
            const double pert=amp*env*(std::sin(2.0*M_PI*x/Lx)*std::cos(2.0*M_PI*y/Ly)
                                      +std::sin(4.0*M_PI*x/Lx)*std::sin(2.0*M_PI*y/Ly));
            V[c]+=static_cast<real_t>(pert);
            W[c]+=static_cast<real_t>(0.5*amp*env*std::sin(2.0*M_PI*x/Lx));
        }
    } else {
        // WARM START — read the source restart (any mesh) + interpolate (staggered,
        // trilinear, x/y periodic, z clamp) onto this run's grid. U,V,W,T,P(=P_acc).
        const double Lx=cfg.get<double>("domain","Lx"), Ly=cfg.get<double>("domain","Ly"), Lz=cfg.get<double>("domain","Lz");
        const double gz=cfg.get_or<double>("mesh_stretch","gamma3",0.0);
        int sdim[3]={0,0,0};
        if (mpi.is_root()) {
            std::ifstream mf(restart_in + "/meta.txt"); std::string k; double v;
            while (mf>>k>>v) { if(k=="n1m")sdim[0]=(int)v; else if(k=="n2m")sdim[1]=(int)v; else if(k=="n3m")sdim[2]=(int)v; }
        }
        MPI_Bcast(sdim,3,MPI_INT,0,topo.cart_comm());
        if (sdim[0]<=0||sdim[1]<=0||sdim[2]<=0) throw std::runtime_error("warm start: bad/missing meta.txt dims in " + restart_in);
        post::SrcMesh sm = post::make_src_mesh(sdim[0],sdim[1],sdim[2], Lx,Ly,Lz, gz);
        std::vector<real_t> buf;
        const core::Var vars[5]={core::Var::U,core::Var::V,core::Var::W,core::Var::T,core::Var::P};
        const char*     names[5]={"U","V","W","T","P"};
        const int       stag[5] ={0,1,2,-1,-1};   // U:x-face V:y-face W:z-face T,P:center

        // I/O path: "slab" (Tier 1, default) reads only this rank's source z-slab via
        // collective MPI-IO (per-rank memory ~ full_field/N_z, no Bcast); "replicate"
        // (legacy) reads the whole field on every rank + Bcast. Both feed the IDENTICAL
        // trilinear interpolation — only the source z indexing differs.
        const std::string ws_io = cfg.get_or<std::string>("restart","warmstart_io","slab");
        const bool use_slab = (ws_io != "replicate");
        int kz0 = 0, slab_n3 = -1;
        if (use_slab) {
            // This rank's target z-extent (interior faces bound every staggered node).
            const int n3t = static_cast<int>(sub.n_total()[2]), h = kHaloWidth;
            const auto& tzf = grid.xf(Direction::Z);
            const double z_min = static_cast<double>(tzf[h]);
            const double z_max = static_cast<double>(tzf[n3t - h]);
            post::compute_src_z_box(sm, z_min, z_max, /*margin=*/2, kz0, slab_n3);
        }
        for (int q=0;q<5;++q) {
            const std::string fpath = restart_in + "/" + std::string(names[q]) + ".bin";
            if (use_slab) post::read_source_slab(buf, sdim[0],sdim[1],sdim[2], kz0, slab_n3, fpath, topo.cart_comm());
            else          post::read_global_field(buf, sdim[0],sdim[1],sdim[2], fpath, topo.cart_comm());
            post::interp_global_to_field(cpu_fields[vars[q]], grid, buf, sm, stag[q], kz0, slab_n3);
        }
        if (mpi.is_root())
            std::printf("[warm-start] interpolated %dx%dx%d -> %dx%dx%d from %s (io=%s)\n",
                        sdim[0],sdim[1],sdim[2], n_global[0],n_global[1],n_global[2],
                        restart_in.c_str(), use_slab ? "slab" : "replicate");

        // dry-run: dump per-var interior checksums (for slab-vs-replicate A/B), then exit.
        if (cfg.get_or<int>("restart","warmstart_dryrun",0)) {
            const auto nt=sub.n_total(); const int n1=nt[0],n2=nt[1],n3=nt[2],h=kHaloWidth;
            if (mpi.is_root()) std::printf("[warm-start dryrun] var: sum  sumsq  min  max  (global interior, io=%s)\n", use_slab?"slab":"replicate");
            for (int q=0;q<5;++q) {
                const real_t* d = cpu_fields[vars[q]].data();
                double s=0, s2=0, mn=1e300, mx=-1e300;
                for (int i=h;i<n1-h;++i) for (int j=h;j<n2-h;++j) for (int k=h;k<n3-h;++k) {
                    const double v=(double)d[((std::size_t)i*n2+j)*n3+k];
                    s+=v; s2+=v*v; mn=std::min(mn,v); mx=std::max(mx,v);
                }
                double gs,gs2,gmn,gmx;
                MPI_Reduce(&s,&gs,1,MPI_DOUBLE,MPI_SUM,0,topo.cart_comm());
                MPI_Reduce(&s2,&gs2,1,MPI_DOUBLE,MPI_SUM,0,topo.cart_comm());
                MPI_Reduce(&mn,&gmn,1,MPI_DOUBLE,MPI_MIN,0,topo.cart_comm());
                MPI_Reduce(&mx,&gmx,1,MPI_DOUBLE,MPI_MAX,0,topo.cart_comm());
                if (mpi.is_root()) std::printf("[warm-start dryrun] %s: %.15e %.15e %.15e %.15e\n", names[q],gs,gs2,gmn,gmx);
            }
            MPI_Barrier(topo.cart_comm());
            return 0;   // RAII (topo, then mpi=MpiContext) frees comm + MPI_Finalize on return
        }
    }
    {
        const auto nt=sub.n_total(); const int n1=nt[0], n2=nt[1], n3=nt[2], h=kHaloWidth;
        real_t* W=cpu_fields[core::Var::W].data();
        const auto& ztopo=sub.topology().axis(Direction::Z);
        if (ztopo.west_rank==MPI_PROC_NULL)
            for(int i=0;i<n1;++i) for(int j=0;j<n2;++j) W[(i*n2+j)*n3+h]=0;
        if (ztopo.east_rank==MPI_PROC_NULL)
            for(int i=0;i<n1;++i) for(int j=0;j<n2;++j) W[(i*n2+j)*n3+(n3-h)]=0;
    }
    core::sync_field_cpu(cpu_fields, core::Var::U, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::V, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::W, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::P, bc, sub);
    core::sync_field_cpu(cpu_fields, core::Var::T, bc, sub);

    // GPU setup
    core::bind_gpu_to_local_rank_gpu(mpi);
    core::GpuGridArrays gpu_grid_arrays(grid);
    domain.gpu_grid = &gpu_grid_arrays;

    core::GpuFields gpu_fields;
    gpu_fields.add(core::Var::U, sub); gpu_fields.add(core::Var::V, sub);
    gpu_fields.add(core::Var::W, sub); gpu_fields.add(core::Var::P, sub);
    gpu_fields.add(core::Var::T, sub);
    gpu_fields.add_constant(core::Const::nu,      static_cast<real_t>(nu));
    gpu_fields.add_constant(core::Const::alpha_T, static_cast<real_t>(alpha_T));

    gpu_fields[core::Var::U].to_device(cpu_fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_device(cpu_fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_device(cpu_fields[core::Var::W].data());
    gpu_fields[core::Var::P].to_device(cpu_fields[core::Var::P].data());
    gpu_fields[core::Var::T].to_device(cpu_fields[core::Var::T].data());

    equation::momentum::GpuLinearSystem gpu_momentum(sub);
    equation::scalar::GpuLinearSystem   gpu_scalar(sub);
    equation::pressure::LinearSystem    pressure;

    // Accumulated pressure Pⁿ (van Kan) lives in Var::P — already seeded by the
    // gpu_fields[Var::P] upload above (0 on cold start, interpolated P on warm-
    // start). The per-step increment δp is owned by the pressure LinearSystem.

    // OB unit property fields for NOB var kernels (mu=1, inv_rho=1, kappa=1,
    // dkappa=0, inv_rhocp=1, dinv_rhocp=0).  With DeltaT=0 the default
    // PropertyModel evaluates to unit constants on every cell → the var kernels
    // reduce to the correct PaScaL_TCS OB scheme with semi-implicit convection
    // and −M·q^n residual correction, fixing the dt collapse at Ra=1e7.
    physics::properties::GpuPropertyFields gpu_props(sub);
    physics::properties::update_properties_gpu(gpu_props, gpu_fields[core::Var::T],
                                   physics::properties::PropertyModel{});

    // Warm start: the interpolated velocity is not discretely divergence-free →
    // one pressure projection cleans it before time marching (methodology step 4).
    if (!restart_in.empty()) {
        const real_t dt0 = static_cast<real_t>(cfg.get_or<double>("time","dt_start",0.001));
        equation::pressure::solve_pressure_constant_property_gpu(domain, bc, gpu_fields, pressure, dt0);
        if (mpi.is_root())
            std::printf("[warm-start] projected interpolated field divergence-free (dt0=%.2e)\n", dt0);
    }

    // GPU statistics accumulator (no full-field D2H during loop)
    GpuDhvcAccum gpu_accum(sub, grid, gpu_grid_arrays);
    post::Stats  stats;
    post::init_statistics_cpu(stats, domain);
    const std::vector<double>& zc_global = stats.zc_global;

    // Monitor CSV
    FILE* mon_fp = nullptr;
    if (mpi.is_root()) {
        mon_fp = std::fopen((dir_stats + "monitor.csv").c_str(), "w");
        if (mon_fp) std::fprintf(mon_fp, "step,t,dt,div,Re_dstar,Re_inst,KE,Nu,theta_rms\n");
    }

    physics::buoyancy::BuoyancyParams buoy;
    buoy.gravity_axis = 0; buoy.coeff=1.0; buoy.T_ref=0.0;
    buoy.a12pera11=0.0; buoy.delta_T=0.0;

    const double total_vol = physics::forcing::channel_total_volume_cpu(domain);
    const auto& ztopo = sub.topology().axis(Direction::Z);
    const bool has_lo = (ztopo.west_rank == MPI_PROC_NULL);
    const bool has_hi = (ztopo.east_rank == MPI_PROC_NULL);

    if (mpi.is_root())
        std::printf("[dhvc_gpu] Ra=%.2e Pr=%.3g nu=%.4e alpha=%.4e n=%dx%dx%d GPU\n",
                    Ra, Pr, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

    // ── time loop ─────────────────────────────────────────────────────────────
    double time = 0.0; int step = 0;
    double net_dpdx = 0.0;
    double t_ss = 0.0; int n_time_steps = 10;
    auto t00 = std::chrono::high_resolution_clock::now();
    using hrc = std::chrono::high_resolution_clock;
    auto ms_since = [](hrc::time_point t) {
        return std::chrono::duration<double,std::milli>(hrc::now()-t).count();
    };
    double ts_cfl=0, ts_scalar=0, ts_mom=0, ts_corr=0, ts_sync=0, ts_pres=0;
    // Whole-run compute timer (never reset, unlike t00) + per-monitor-interval rate.
    MPI_Barrier(topo.cart_comm());
    const auto t_loop_start = hrc::now();
    auto t_last_mon = t_loop_start;
    int  last_mon_step = 0;

    while (time < t_end && step < n_steps) {
        const bool do_timing = (step < n_time_steps);
        hrc::time_point tPrev;

        if (do_timing) { cudaDeviceSynchronize(); tPrev = hrc::now(); }
        const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);
        if (do_timing) { cudaDeviceSynchronize(); ts_cfl += ms_since(tPrev); tPrev = hrc::now(); }

        // (1) Scalar: T^n → T^{n+1}  (NOB var path with unit props = PaScaL_TCS scheme)
        equation::scalar::solve_scalar_variable_property_gpu(domain, bc, gpu_fields,
            gpu_props.KPP, gpu_props.dKPP, gpu_props.invRhoCp, gpu_props.dinvRhoCp,
            gpu_scalar, dt);
        core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);
        if (do_timing) { cudaDeviceSynchronize(); ts_scalar += ms_since(tPrev); tPrev = hrc::now(); }

        // (2) Momentum predictor + OB buoyancy  (NOB var path with unit props)
        equation::momentum::assemble_momentum_variable_property_gpu(domain, gpu_fields,
            gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
        physics::buoyancy::add_buoyancy_force_constant_property_gpu(domain, gpu_momentum, gpu_fields[core::Var::T], buoy, dt);
        equation::pressure::add_pressure_gradient_constant_property_gpu(domain, gpu_momentum, gpu_fields[core::Var::P], dt);  // incremental: −dt·∇Pⁿ
        equation::momentum::solve_momentum_variable_property_gpu(domain, bc, gpu_fields,
            gpu_props.Mu, gpu_props.invRho, gpu_momentum, dt);
        equation::momentum::update_velocity_gpu(gpu_fields, gpu_momentum);
        if (do_timing) { cudaDeviceSynchronize(); ts_mom += ms_since(tPrev); tPrev = hrc::now(); }

        // (3) Zero net streamwise flux
        physics::forcing::apply_mass_flow_correction_gpu(domain, gpu_fields, 0.0, double(dt), total_vol, net_dpdx);
        if (do_timing) { cudaDeviceSynchronize(); ts_corr += ms_since(tPrev); tPrev = hrc::now(); }

        core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);
        core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
        core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);

        // (4) Zero W at wall faces (GPU kernel, no D2H)
        {
            const auto nt=sub.n_total();
            const int n1=nt[0], n2=nt[1], n3=nt[2], h=kHaloWidth;
            const int block=256, grid_xy=((n1-2*h)*(n2-2*h)+block-1)/block;
            k_zero_w_wall<<<grid_xy,block>>>(gpu_fields[core::Var::W].data(),
                                             n1, n2, n3, h, has_lo, has_hi);
            core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
        }
        if (do_timing) { cudaDeviceSynchronize(); ts_sync += ms_since(tPrev); tPrev = hrc::now(); }

        // (5) Pressure
        equation::pressure::solve_pressure_constant_property_gpu(domain, bc, gpu_fields, pressure, dt);
        equation::pressure::accumulate_pressure_gpu(gpu_fields, pressure);  // incremental: Pⁿ⁺¹ = Pⁿ + δp
        if (do_timing) { cudaDeviceSynchronize(); ts_pres += ms_since(tPrev); }

        time += double(dt); ++step;

        // (6) GPU statistics accumulation — NO D2H
        if (time >= t_stat_start)
            gpu_accum_dhvc(gpu_accum, gpu_fields, dt);   // every step, dt-weighted

        // (7) Monitor: ~1.6 KB D2H, write monitor.csv (+ live wall-clock / ms-per-step)
        if (step % nmonitor == 0) {
            cudaDeviceSynchronize();                          // all step work done → wall is exact
            const double cum_wall_s     = ms_since(t_loop_start) / 1000.0;
            const double recent_ms_step = ms_since(t_last_mon) / std::max(1, step - last_mon_step);
            gpu_monitor_dhvc(gpu_accum, gpu_fields, step, time, dt, nu, alpha_T, Ra,
                             mpi.is_root(), topo.cart_comm(), mon_fp, zc_global,
                             cum_wall_s, recent_ms_step);
            t_last_mon = hrc::now(); last_mon_step = step;
        }

        // (8) Stats: cumulative-average profile file-out every t_stat_out of PHYSICAL
        //     TIME from t_stat_start, so QoI convergence is trackable from the .dat seq.
        if (time >= t_next_out && gpu_accum.t_accum > 0.0) {
            gpu_write_stats_dhvc(gpu_accum, stats,
                dir_stats + "stats_t" + std::to_string((long)std::llround(time)) + ".dat",
                step, nu, domain);
            t_next_out += t_stat_out;
        }

        // Timing: first 10 steps → per-step average + estimated total runtime
        if (step == n_time_steps && mpi.is_root()) {
            const double ms_per_step = ms_since(t00) / n_time_steps;
            const double est_total_h = ms_per_step * (t_end / std::max(double(dt), 1e-9)) / 3600000.0;
            std::printf("[TIMING] %d steps  avg=%.1f ms/step\n", n_time_steps, ms_per_step);
            std::printf("[TIMING_BREAKDOWN] per step:  cfl=%.1f  scalar=%.1f  mom=%.1f  corr=%.1f  sync=%.1f  pres=%.1f  ms\n",
                        ts_cfl/n_time_steps, ts_scalar/n_time_steps, ts_mom/n_time_steps,
                        ts_corr/n_time_steps, ts_sync/n_time_steps, ts_pres/n_time_steps);
            std::printf("[TIMING] t_end=%.0f  rough total: %.2f h (n_steps cap=%d)\n",
                        t_end, est_total_h, n_steps);
            std::printf("[TIMING] stats out: from t=%.1f every t=%.1f (to t_end=%.0f)\n",
                        t_stat_start, t_stat_out, t_end);
            t00 = hrc::now(); t_ss = 0.0;
        }
    }

    if (step > n_time_steps && mpi.is_root())
        std::printf("[TIMING_SS] %d steps (steady-state)  total=%.1f ms/step\n",
                    step - n_time_steps, ms_since(t00)/(step - n_time_steps));

    // ── total compute time for the whole run (loop start → here) ─────────────────
    cudaDeviceSynchronize();
    MPI_Barrier(topo.cart_comm());
    const double total_s = ms_since(t_loop_start) / 1000.0;
    if (mpi.is_root()) {
        const double avg_ms = total_s * 1000.0 / std::max(1, step);
        std::printf("[TIMING_TOTAL] compute = %.1f s (%.2f h)  |  %d steps  |  avg %.2f ms/step\n",
                    total_s, total_s / 3600.0, step, avg_ms);
    }

    if (mon_fp) std::fclose(mon_fp);

    // ── checkpoint write (restart file at run end, t=t_end) ──────────────────────
    // U,V,W,T + P (= accumulated pressure P_acc for the incremental scheme). Global
    // C-order MPI-IO (decomposition-independent). meta.txt carries t/step/mesh/Ra for
    // continuation + the (future) cross-mesh interpolation warm-start.
    if (!restart_out.empty()) {
        gpu_fields[core::Var::U].to_host(cpu_fields[core::Var::U].data());
        gpu_fields[core::Var::V].to_host(cpu_fields[core::Var::V].data());
        gpu_fields[core::Var::W].to_host(cpu_fields[core::Var::W].data());
        gpu_fields[core::Var::T].to_host(cpu_fields[core::Var::T].data());
        gpu_fields[core::Var::P].to_host(cpu_fields[core::Var::P].data());   // P.bin = accumulated pressure
        std::error_code ec; std::filesystem::create_directories(restart_out, ec);
        MPI_Barrier(topo.cart_comm());
        post::write_restart_gathered_cpu(cpu_fields[core::Var::U], sub, restart_out + "/U.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::V], sub, restart_out + "/V.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::W], sub, restart_out + "/W.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::T], sub, restart_out + "/T.bin");
        post::write_restart_gathered_cpu(cpu_fields[core::Var::P], sub, restart_out + "/P.bin");
        if (mpi.is_root()) {
            std::ofstream mf(restart_out + "/meta.txt");
            mf << std::setprecision(17)
               << "step " << step << "\n" << "time " << time << "\n"
               << "n1m " << n_global[0] << "\n" << "n2m " << n_global[1] << "\n"
               << "n3m " << n_global[2] << "\n" << "Ra " << Ra << "\n";
            std::printf("[restart] wrote %s  (step=%d t=%.4f; U/V/W/T/P + meta)\n",
                        restart_out.c_str(), step, time);
        }
    }

    const double re_dstar = post::compute_re_delta_star_cpu(stats, nu, domain);
    if (mpi.is_root())
        std::printf("[dhvc_gpu] done: Ra=%.2e  Re_dstar=%.4f  (target=%.4f)\n",
                    Ra, re_dstar, 0.23*std::pow(Ra, 0.28));
    return 0;
}
