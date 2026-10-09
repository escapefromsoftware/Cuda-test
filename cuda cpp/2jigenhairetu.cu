#include <cstdio>
#include <cuda_runtime.h>

__device__ int gpuArray[3][4];

__global__ void process()
{
    gpuArray[1][2] = 10;
    printf("%d\n", gpuArray[1][2]);
}

int main()
{
    process<<<1, 1>>>();
    cudaDeviceSynchronize();
}