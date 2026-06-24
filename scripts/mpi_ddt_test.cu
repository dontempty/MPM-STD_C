// Does CUDA-aware MPI host-stage DERIVED-DATATYPE (non-contiguous) collectives on
// device pointers? PaScaL_TDMA uses MPI_Ialltoallw with strided ddtype. Contiguous
// Alltoall was D2D (cuda-aware OK); this tests a STRIDED MPI_Type_vector Alltoallw.
// If H2D/D2H >> 0 here → derived-datatype path host-stages = the dhvc storm source.
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    const int nblk = (argc>1)?atoi(argv[1]):512;    // # strided blocks
    const int blen = (argc>2)?atoi(argv[2]):8;      // block length (doubles)
    const int strd = (argc>3)?atoi(argv[3]):32;     // stride (doubles)
    const int nstep= (argc>4)?atoi(argv[4]):50;
    // strided datatype (like PaScaL_TDMA's reduced-system gather)
    MPI_Datatype vec; MPI_Type_vector(nblk, blen, strd, MPI_DOUBLE, &vec); MPI_Type_commit(&vec);

    const size_t buf_elems = (size_t)nblk*strd*size + 1024;
    double *sbuf=nullptr,*rbuf=nullptr;
    cudaMalloc(&sbuf, buf_elems*sizeof(double));
    cudaMalloc(&rbuf, buf_elems*sizeof(double));
    cudaMemset(sbuf,0,buf_elems*sizeof(double)); cudaDeviceSynchronize();

    int *cnt = (int*)malloc(size*sizeof(int));
    int *dsp = (int*)malloc(size*sizeof(int));
    MPI_Datatype *typ=(MPI_Datatype*)malloc(size*sizeof(MPI_Datatype));
    MPI_Aint lb,ext; MPI_Type_get_extent(vec,&lb,&ext);
    for(int p=0;p<size;++p){ cnt[p]=1; dsp[p]=(int)(p*ext); typ[p]=vec; }

    // warmup
    MPI_Alltoallw(sbuf,cnt,dsp,typ, rbuf,cnt,dsp,typ, MPI_COMM_WORLD);
    cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);
    for(int s=0;s<nstep;++s){
        MPI_Alltoallw(sbuf,cnt,dsp,typ, rbuf,cnt,dsp,typ, MPI_COMM_WORLD);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("[done] %d device-ptr MPI_Alltoallw (STRIDED ddt: %d blk x %d, stride %d)\n",
                       nstep,nblk,blen,strd);
    MPI_Finalize(); return 0;
}
