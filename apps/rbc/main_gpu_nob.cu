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

// GPU kernel: clamp T to [-0.5, 0.5] (replaces D2H+CPU clamp+H2D each step)
__global__ void k_clip_T_gpu(double* T, std::size_t n) {
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double v = T[idx];
  T[idx] = v < -0.5 ? -0.5 : (v > 0.5 ? 0.5 : v);
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

// GPU accumulate: compute tc and wt on device, D2H 16 bytes, MPI_Allreduce → update stats.
inline void gpu_accumulate_rbc(GpuRbcAccum& ga, rbc::RbcStats& s,
    const core::GpuFields& gf, MPI_Comm comm)
{
  const double* T = gf[core::Var::T].data();
  const double* W = gf[core::Var::W].data();
  cudaMemset(ga.d_tmp2, 0, 2*sizeof(double));
  // k_accum_tc_wt: block (8,8,4)=256, grid covers (nx,ny,nz)
  const dim3 blk(8,8,4);
  const dim3 grd((ga.nx+7)/8,(ga.ny+7)/8,(ga.nz+3)/4);
  k_accum_tc_wt_rbc<<<grd,blk,2*256*sizeof(double)>>>(
      T, W, ga.d_dx3, ga.d_tmp2,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h, ga.kc_local);
  // k_accum_z_profile: 256 threads, grid (ceil(nx*ny/256), nz)
  const int flat_blocks = (ga.nx*ga.ny+255)/256;
  k_accum_z_profile_rbc<<<dim3(flat_blocks,ga.nz),256,2*256*sizeof(double)>>>(
      T, W, ga.d_t_sum_z, ga.d_w2_sum_z,
      ga.nx, ga.ny, ga.nz, ga.n2t, ga.n3t, ga.h);
  cudaDeviceSynchronize();
  // D2H: 16 bytes
  double local[2]; cudaMemcpy(local, ga.d_tmp2, 2*sizeof(double), cudaMemcpyDeviceToHost);
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, comm);
  s.wt_sum += global[0];
  s.tc_sum += global[1] / (double(ga.nxg)*ga.nyg);
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
  if (!is_root) return;
  const double dv=global[5], umax=global[0], vmax=global[1], wmax=global[2];
  const double Tmax=global[3], Tmin=-global[4];
  const double tc  = (s.tc_n>0) ? s.tc_sum/s.tc_n : 0.0;
  const double nu_val = (s.tc_n>0) ? 1.0 + (s.wt_sum/(double(ga.nxg)*ga.nyg)/s.tc_n)/alpha_T : 0.0;
  std::printf("step=%d t=%.3g dt=%.3g div=%.2e Theta_c=%.4f | umax=%.3g vmax=%.3g wmax=%.3g T=[%.3f,%.3f]\n",
              step, time, double(dt), dv, tc, umax, vmax, wmax, Tmin, Tmax);
  if (s.tc_n > 0)
    std::printf("  [Nu] running: Nu=%.4f (tc_n=%ld)\n", nu_val, s.tc_n);
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

  if (mpi.is_root())
    std::printf("[rbc_gpu_nob] Ra=%.2e Pr=%.4g DeltaT=%.1f nu=%.4e alpha=%.4e  n=%dx%dx%d  (GPU hybrid)\n",
                Ra, Pr, DeltaT, nu, alpha_T, n_global[0], n_global[1], n_global[2]);

  // ── time loop ─────────────────────────────────────────────────────────────
  double time = rmeta.time; int step = rmeta.step;
  rbc::RbcStats stats(sub.n_global()[2]);
  rbc::apply_stats_from_restart(stats, rmeta);
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

  while (time < t_end && step < n_steps) {
    auto t00 = hrc::now();

    const real_t dt = driver::compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap);
    physics::update_properties_gpu(gpu_props, gpu_fields[core::Var::T], pmodel);

    cudaDeviceSynchronize();
    auto t_s0 = hrc::now();
    equation::assemble_scalar_var_diff_gpu(domain, gpu_fields, gpu_props.kappa, gpu_props.dkappa,
                                           gpu_props.inv_rhocp, gpu_props.dinv_rhocp, gpu_scalar, dt);
    equation::solve_scalar_var_gpu(domain, bc, gpu_fields, gpu_props.kappa, gpu_props.dkappa,
                                   gpu_props.inv_rhocp, gpu_props.dinv_rhocp, gpu_scalar, dt);
    core::sync_field_gpu(gpu_fields, core::Var::T, bc, sub);
    cudaDeviceSynchronize();
    t_scalar += ms_since(t_s0);

    // Re-evaluate properties from the just-solved T^{n+1} so momentum viscosity/density
    // (mu, inv_rho) and the NOB buoyancy inv_rho use T^{n+1} — matches PaScaL_TCS
    // momentum_coeffi(T) called AFTER thermal_solver (MPM-STD staggered-time scheme).
    // Energy above already used the T^n properties from line 408.
    physics::update_properties_gpu(gpu_props, gpu_fields[core::Var::T], pmodel);

    auto t_m0 = hrc::now();
    equation::assemble_momentum_var_visc_gpu(domain, gpu_fields, gpu_props.mu, gpu_props.inv_rho, gpu_momentum, dt);
    physics::add_buoyancy_force_nob_gpu(gpu_momentum, gpu_fields[core::Var::T], gpu_props.inv_rho, buoy, dt);
    equation::solve_momentum_var_gpu(domain, bc, gpu_fields, gpu_props.mu, gpu_props.inv_rho, gpu_momentum, dt);
    equation::update_velocity_gpu(gpu_fields, gpu_momentum);
    core::sync_field_gpu(gpu_fields, core::Var::U, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::V, bc, sub);
    core::sync_field_gpu(gpu_fields, core::Var::W, bc, sub);
    cudaDeviceSynchronize();
    t_mom += ms_since(t_m0);

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

    // G4: GPU pressure solve (NOB — no H↔D transfers for pressure)
    // Report/25: faithful NOB pressure restored. ExtraTerm (Poisson) and the
    // (invRhoc-1)*∇dPhat velocity term are a cancelling PAIR (div=0 exact), not
    // double-counting; the 2026-07-02 OB swap was a misdiagnosis (θ_c unchanged).
    auto t_p0 = hrc::now();
    equation::solve_pressure_nob_gpu(domain, bc, gpu_fields, pressure, dt, gpu_props.inv_rho);
    cudaDeviceSynchronize();
    t_pressure += ms_since(t_p0);

    // T clipping: GPU kernel — no D2H/H2D
    auto t_c0 = hrc::now();
    if (DeltaT > 0.0) {
      const std::size_t nf = (std::size_t)sub.n_total()[0]*sub.n_total()[1]*sub.n_total()[2];
      k_clip_T_gpu<<<(int)((nf+255)/256), 256>>>(gpu_fields[core::Var::T].data(), nf);
    }
    cudaDeviceSynchronize();
    t_clip += ms_since(t_c0);

    time += double(dt); ++step;

    // GPU statistics: device reductions only — no field D2H (≤48 bytes per step)
    auto t_d0 = hrc::now();
    if (step >= nstat_start)
      gpu_accumulate_rbc(gpu_accum, stats, gpu_fields, topo.cart_comm());
    t_diag += ms_since(t_d0);

    auto t_a0 = hrc::now();
    if (step % nmonitor == 0)
      gpu_print_step_rbc(gpu_accum, stats, gpu_fields, step, time, dt, alpha_T, mpi.is_root(), topo.cart_comm(), mon_fp);
    t_accum += ms_since(t_a0);

    if (step >= nstat_start && nstat_write > 0 && step % nstat_write == 0) {
      const std::string sp = dir_stats + "stats_" + std::to_string(step) + ".dat";
      gpu_write_stats_rbc(gpu_accum, stats, sp, step, Ra, DeltaT,
                          mpi.is_root(), topo.cart_comm(), grid, sub);
    }

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

  // Write restart files (U/V/W/T + stats.txt) for continuation runs
  if (!restart_out.empty()) {
    gpu_fields[core::Var::U].to_host(fields[core::Var::U].data());
    gpu_fields[core::Var::V].to_host(fields[core::Var::V].data());
    gpu_fields[core::Var::W].to_host(fields[core::Var::W].data());
    gpu_fields[core::Var::T].to_host(fields[core::Var::T].data());
    rbc::write_restart(fields, sub, stats, step, time, restart_out, topo.cart_comm());
  }
  return 0;
}
