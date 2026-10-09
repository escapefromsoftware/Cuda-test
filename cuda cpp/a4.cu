#include <cstdio>
#include <cuda_runtime.h>

__global__ void doubleArray(int* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] *= 2;
    }
}

int main()
{
    const int N = 10;

    int h_a[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }

    int* d_a;

    cudaMalloc(&d_a, N * sizeof(int));

    cudaMemcpy(
        d_a,
        h_a,
        N * sizeof(int),
        cudaMemcpyHostToDevice
    );

    doubleArray<<<1, 16>>>(d_a, N);

    cudaMemcpy(
        h_a,
        d_a,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < N; i++) {
        printf("%d ", h_a[i]);
    }

    cudaFree(d_a);

    return 0;
}