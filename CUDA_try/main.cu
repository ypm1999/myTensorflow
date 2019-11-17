#include <bits/stdc++.h>
#include <cuda/cuda_runtime.h>
#define float double
using namespace std;

const int N = 10000, K = 5000, M = 10000;


const int THREAD_NUM = 256;
const int BLOCK_NUM = 32;
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
        cerr <<  "There is no device." << endl;
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
        cerr << "There is no device supporting CUDA 1.x." << endl;
        return false;
    }
    cudaSetDevice(i);
    have_init = true;
    printf("CUDA initialized.\n");
    return true;
}

__global__ static void matmulKernel(float *A, float *B, float *C, int n, int m, int k) {
    const int pos = threadIdx.x + THREAD_NUM * blockIdx.x;
    for(int i = pos; i < n * m; i += n * m){
        const int x = pos / m, y = pos % m;
        float Csub = 0, loss = 0, tmp = 0;
        for(int i = 0; i < k; i++){
            tmp = A[x * k + i] * B[i * m + y];
            loss += tmp + Csub - tmp - Csub;
            Csub += tmp;
        }
        C[x * m + y] = Csub + loss;
    }
}


void matmul(float* A, float* B, float* C, int n, int m, int k){
    if(!have_init){
        if(!InitCUDA())
            return;
    }

    float *Ad = nullptr, *Bd = nullptr, *Cd = nullptr;

    cudaMalloc((void**)&Ad, sizeof(float) * n * k);
    cudaMalloc((void**)&Bd, sizeof(float) * k * m);
    cudaMalloc((void**)&Cd, sizeof(float) * n * m);

    cudaMemcpy(Ad, A, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, sizeof(float) * k * m, cudaMemcpyHostToDevice);

    matmulKernel <<<BLOCK_NUM, THREAD_NUM, 0 >>>(Ad, Bd, Cd, N, M, K);

    cudaMemcpy(B, Bd, sizeof(float)* k * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, Cd, sizeof(float)* n * m, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

}

float A[N][K], B[K][M];
float C[N][M];

int main(){

    for(int i = 0; i < N; i++)
        for(int j = 0; j < K; j++)
            A[i][j] = 1;

    for(int i = 0; i < K; i++)
        for(int j = 0; j < M; j++)
            B[i][j] = 1;
    clock_t st = clock();
    matmul(A[0], B[0], C[0], N, M, K);

    fprintf(stderr, "Total time:%.3lf\n",  1.0 * (clock() - st) / CLOCKS_PER_SEC);
    return 0;
}
