#include <cstdio>
#include <cuda_runtime.h>

__global__ void practice(){
    int i = threadIdx.x;
    printf("Hello from thread %d\n", i * blockDim.x + blockIdx.x * blockDim.x);
}

int main() {
    practice<<<3, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}