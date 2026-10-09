#include<cstdio>
#include<cuda_runtime.h>

__global__ void showThreadID()
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Block ID: %d, Thread ID: %d global: %d\n", blockIdx.x, threadIdx.x, threadId);
}

int main()
{
    int numBlocks = 2;
    int threadsPerBlock = 4;

    showThreadID<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}