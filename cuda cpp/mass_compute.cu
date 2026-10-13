#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(error));                            \
            return 1;                                                           \
        }                                                                       \
    } while (false)

__host__ __device__ std::uint32_t calculate(std::uint32_t value, int rounds)
{
    for (int round = 0; round < rounds; ++round) {
        value = value * 1664525u + 1013904223u;
        value ^= value >> 16;
    }
    return value;
}

__global__ void calculateMany(std::uint32_t* output, std::size_t count, int rounds)
{
    const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;
    for (std::size_t index = first; index < count; index += stride) {
        output[index] = calculate(static_cast<std::uint32_t>(index + 1), rounds);
    }
}

bool parsePositive(const char* text, unsigned long long limit,
                   unsigned long long& value)
{
    if (*text == '\0' || *text == '-') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed == 0 || parsed > limit) {
        return false;
    }
    value = parsed;
    return true;
}

int main(int argc, char** argv)
{
    if (argc > 3) {
        std::fprintf(stderr, "Usage: mass_compute.exe [elements] [rounds]\n");
        return 1;
    }

    unsigned long long elements = 1ull << 20;
    unsigned long long rounds = 256;
    if ((argc >= 2 && !parsePositive(argv[1], 1ull << 24, elements)) ||
        (argc >= 3 && !parsePositive(argv[2], 4096, rounds))) {
        std::fprintf(stderr, "elements: 1..16777216, rounds: 1..4096\n");
        return 1;
    }

    const std::size_t count = static_cast<std::size_t>(elements);
    const std::size_t bytes = count * sizeof(std::uint32_t);
    std::vector<std::uint32_t> gpuResult(count);
    std::vector<std::uint32_t> cpuResult(count);

    std::uint32_t* deviceResult = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceResult, bytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int threadsPerBlock = 256;
    const int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
    calculateMany<<<blocks, threadsPerBlock>>>(deviceResult, count,
                                               static_cast<int>(rounds));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto gpuStart = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    calculateMany<<<blocks, threadsPerBlock>>>(deviceResult, count,
                                               static_cast<int>(rounds));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernelMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMilliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), deviceResult, bytes,
                          cudaMemcpyDeviceToHost));
    const auto gpuEnd = std::chrono::steady_clock::now();

    const auto cpuStart = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < count; ++index) {
        cpuResult[index] = calculate(static_cast<std::uint32_t>(index + 1),
                                     static_cast<int>(rounds));
    }
    const auto cpuEnd = std::chrono::steady_clock::now();

    for (std::size_t index = 0; index < count; ++index) {
        if (gpuResult[index] != cpuResult[index]) {
            std::fprintf(stderr, "Mismatch at %zu: GPU=%u CPU=%u\n", index,
                         gpuResult[index], cpuResult[index]);
            return 1;
        }
    }

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::printf("Elements: %zu, rounds per element: %llu\n", count, rounds);
    std::printf("GPU kernel: %.3f ms\n", kernelMilliseconds);
    std::printf("GPU launch + result copy: %.3f ms\n",
                milliseconds(gpuStart, gpuEnd));
    std::printf("CPU calculation: %.3f ms\n", milliseconds(cpuStart, cpuEnd));
    std::printf("Verification: OK (all %zu results match)\n", count);

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceResult));
    return 0;
}
