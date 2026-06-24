// cuFFT microbenchmark — replicate the pressure engine's per-slab Y-FFT loop
// (plan_y_ at pressure_engine_gpu.cuh:574) to test whether cuFFT itself issues
// H2D/D2H (cuMemcpyAsync) per exec. NO solver, NO MPI, NO halo — pure cuFFT.
// If this alone shows ~ (n3_I * 6) H2D/D2H per "step", cuFFT is the storm source.
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);abort();} }while(0)
#define FK(x) do{ cufftResult r=(x); if(r!=CUFFT_SUCCESS){printf("CUFFT err %d @%d\n",(int)r,__LINE__);abort();} }while(0)

int main(int argc, char** argv) {
    // 1e6 z-split (np3=2) representative pencil sizes:
    const int N2g   = (argc>1)?atoi(argv[1]):192;   // y global (FFT length)
    const int h1p   = (argc>2)?atoi(argv[2]):193;   // x-freq this rank (batch & stride)
    const int n3_I  = (argc>3)?atoi(argv[3]):48;    // z-slabs per rank (loop count)
    const int nstep = (argc>4)?atoi(argv[4]):50;    // "time steps"
    const size_t slab = (size_t)h1p * N2g;          // one z-slab (complex elems)

    printf("microbench: N2g=%d h1p=%d n3_I=%d nstep=%d slab=%zu | %d FFT exec/step\n",
           N2g, h1p, n3_I, nstep, slab, n3_I);

    // plan_y_ : batched 1D Z2Z, n=N2g, istride=h1p, idist=1, batch=h1p  (per slab)
    cufftHandle plan;
    int n[1] = { N2g };
    FK(cufftPlanMany(&plan, 1, n, n, h1p, 1, n, h1p, 1, CUFFT_Z2Z, h1p));

    cufftDoubleComplex* d_hat = nullptr;
    CK(cudaMalloc(&d_hat, (size_t)n3_I * slab * sizeof(cufftDoubleComplex)));
    CK(cudaMemset(d_hat, 0, (size_t)n3_I * slab * sizeof(cufftDoubleComplex)));
    CK(cudaDeviceSynchronize());

    // warm up (exclude plan/first-call init from the measured region)
    for (int k=0;k<n3_I;++k) FK(cufftExecZ2Z(plan, d_hat+(size_t)k*slab, d_hat+(size_t)k*slab, CUFFT_FORWARD));
    CK(cudaDeviceSynchronize());

    // measured: nstep × (n3_I fwd + n3_I inv)  — exactly the pressure fwd+bwd Y-FFT loops
    for (int s=0;s<nstep;++s) {
        for (int k=0;k<n3_I;++k) FK(cufftExecZ2Z(plan, d_hat+(size_t)k*slab, d_hat+(size_t)k*slab, CUFFT_FORWARD));
        for (int k=0;k<n3_I;++k) FK(cufftExecZ2Z(plan, d_hat+(size_t)k*slab, d_hat+(size_t)k*slab, CUFFT_INVERSE));
    }
    CK(cudaDeviceSynchronize());
    printf("done: %d steps × %d exec = %d cufftExecZ2Z\n", nstep, 2*n3_I, nstep*2*n3_I);
    cufftDestroy(plan); cudaFree(d_hat);
    return 0;
}
