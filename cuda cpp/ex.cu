#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloFromGpu()
{
    printf("MANKO!\n");
}

int main()
{
    helloFromGpu<<<3, 2>>>();

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}
