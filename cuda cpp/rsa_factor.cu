#include "rsa_common.cuh"

#include <cmath>

__global__ void findFactor(std::uint64_t modulus, std::uint64_t root,
                           unsigned long long* smallest)
{
    const std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint64_t divisor = 3 + 2 * index;
    if (divisor <= root && modulus % divisor == 0) {
        atomicMin(smallest, static_cast<unsigned long long>(divisor));
    }
}

int main()
{
    constexpr std::uint64_t p = 1000003;
    constexpr std::uint64_t q = 1000033;
    constexpr std::uint64_t modulus = p * q;
    auto root = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(modulus)));
    while (root + 1 <= modulus / (root + 1)) {
        ++root;
    }
    while (root > modulus / root) {
        --root;
    }
    const auto candidateCount = (root - 3) / 2 + 1;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((candidateCount + threads - 1) / threads);
    unsigned long long* deviceFactor = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceFactor, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(deviceFactor, 0xff, sizeof(unsigned long long)));
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    findFactor<<<blocks, threads>>>(modulus, root, deviceFactor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    unsigned long long factor = 0;
    CUDA_CHECK(cudaMemcpy(&factor, deviceFactor, sizeof(factor), cudaMemcpyDeviceToHost));
    if (factor != p || modulus / factor != q) {
        std::fprintf(stderr, "Factorization verification failed\n");
        return 1;
    }
    std::printf("Toy factorization: %llu = %llu x %llu\n", modulus, factor,
                modulus / factor);
    std::printf("Tested %llu odd divisors in %.3f ms on GPU. Verification: OK\n",
                candidateCount, milliseconds);
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceFactor));
    return 0;
}
