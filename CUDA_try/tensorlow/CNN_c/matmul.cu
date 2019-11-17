#include <cstdio>
#include <iostream>
#include <cuda/cuda_runtime.h>
using namespace std;
const int BLOCK_SIZE = 24;
bool have_init = false;


void printDeviceProp(const cudaDeviceProp &prop) {
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", (int)prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", (int)prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
//    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", (int)prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", (int)prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
//    printf("textureAlignment : %d.\n", (int)prop.textureAlignment);
//    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
//    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA() {
    int count = 0;
    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for (i = 0; i < count; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //打印设备信息
        printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
            if (prop.major >= 1) {
                break;
            }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    have_init = true;

    printf("CUDA initialized.\n");
    return true;
}

__global__ static void matmulKernel(float *A, size_t lda, float *B, size_t ldb, float *C, size_t ldc,
                                    float beta, size_t n) {
    __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE], matB[BLOCK_SIZE][BLOCK_SIZE];
    const int tidr = threadIdx.x;
    const int tidc = threadIdx.y;
    const int bidr = blockIdx.x * BLOCK_SIZE;
    const int bidc = blockIdx.y * BLOCK_SIZE;
    int i, j;
    float result = beta * C[(tidr + bidr) * ldc + tidc + bidc], loss = 0, tmp, t;
    for(j = 0; j < n; j += BLOCK_SIZE){
        matA[tidr][tidc] = A[(tidr + bidr) * lda + tidc + j];
        matB[tidr][tidc] = B[(tidr + j) * ldb + tidc + bidc];

        __syncthreads();

        for(i = 0; i < BLOCK_SIZE; i++){
            loss -= matA[tidr][i] * matB[i][tidc];
            tmp = result - loss;
            loss = (tmp - result) + loss;
            result = tmp;
        }

        __syncthreads();
    }
    C[(tidr + bidr) * ldc + tidc + bidc] = result;
}

extern "C" void out(float* a, int n, int m){
    for(int i  = 0; i < n; i++, puts(""))
        for(int j = 0; j < m; j++)
            printf("%.2f ", a[i * m + j]);
    puts("--------------------------------");
}

double Time = 0.0;
int cnt = 0;
//n * lda
// * ldb(ｎ)
extern "C" void matmul_gpu(float* A, float* B, float* C, int n, int m, int k, float beta){
    if(!have_init){
        if(!InitCUDA())
            return;
    }

    float *Ad, *Bd, *Cd;
    int nn = (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int mm = (m + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int kk = (k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    size_t lda = (size_t)k, ldb = (size_t)m, ldc = ldb;
    size_t pitch_a, pitch_b, pitch_c;
    clock_t st = clock();
    cudaMallocPitch((void**)&Ad, &pitch_a, sizeof(float) * kk, nn);
    cudaMallocPitch((void**)&Bd, &pitch_b, sizeof(float) * mm, kk);
    cudaMallocPitch((void**)&Cd, &pitch_c, sizeof(float) * mm, nn);
    cudaMemset2D(Ad, pitch_a, 0, sizeof(float) * kk, nn);
    cudaMemset2D(Bd, pitch_b, 0, sizeof(float) * mm, kk);

    if(A != NULL)
        cudaMemcpy2D(Ad, pitch_a, A, sizeof(float) * lda, sizeof(float) * lda, n, cudaMemcpyHostToDevice);
    if(B != NULL)
        cudaMemcpy2D(Bd, pitch_b, B, sizeof(float) * ldb, sizeof(float) * ldb, k, cudaMemcpyHostToDevice);
    if(beta != 0){
        cudaMemset2D(Cd, pitch_c, 0, sizeof(float) * mm, nn);
        cudaMemcpy2D(Cd, pitch_c, C, sizeof(float) * ldc, sizeof(float) * ldc, n, cudaMemcpyHostToDevice);
    }


    dim3 blocks(nn / BLOCK_SIZE, mm / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//    printf("%d %d %d %d %d %d %d %d\n", n, m, k, blocks.x, blocks.y, nn, mm, kk);
    matmulKernel <<<blocks, threads>>>(Ad, pitch_a / 4, Bd, pitch_b / 4, Cd, pitch_c / 4, beta, k);
    cudaMemcpy2D(C, sizeof(float) * ldc, Cd, pitch_c, sizeof(float) * ldc, n, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    Time += clock() - st;
    cnt++;
    if(cnt % 20 == 0)
        fprintf(stderr, "time:%.3lf\n", Time / CLOCKS_PER_SEC);
}

