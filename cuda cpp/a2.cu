#include <cstdio>
#include <cuda_runtime.h>

__device__ int gpuArray[8];

__global__ void process()
{
    int i = threadIdx.x;
    gpuArray[i] = i * i;

    printf("gpuArray[%d] = %d\n", i, gpuArray[i]);
}

int main()
{
    process<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}