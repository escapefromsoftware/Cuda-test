#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
            return 1;                                                           \
        }                                                                       \
    } while (false)

constexpr int tileSize = 16;

// C = A x B。1スレッドがCの1要素を計算する。
__global__ void multiplyMatrices(const float* a, const float* b, float* c, int size)
{
    __shared__ float tileA[tileSize][tileSize];
    __shared__ float tileB[tileSize][tileSize];
    const int row = blockIdx.y * tileSize + threadIdx.y;
    const int col = blockIdx.x * tileSize + threadIdx.x;
    float sum = 0.0f;

    // 行列を16 x 16ずつ共有メモリに読み、ブロック内で使い回す。
    for (int base = 0; base < size; base += tileSize) {
        const int aCol = base + threadIdx.x;
        const int bRow = base + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (row < size && aCol < size) ? a[row * size + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < size && col < size) ? b[bRow * size + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tileSize; ++k) {
            sum = fmaf(tileA[threadIdx.y][k], tileB[k][threadIdx.x], sum);
        }
        __syncthreads();
    }
    if (row < size && col < size) {
        c[row * size + col] = sum;
    }
}

bool parsePositive(const char* text, int limit, int& value)
{
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' || parsed < 1 || parsed > limit) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void printCorner(const char* name, const std::vector<float>& matrix, int size)
{
    const int shown = std::min(size, 4);
    std::printf("%s (%s%d x %d):\n", name,
                size > shown ? "top-left " : "", shown, shown);
    for (int row = 0; row < shown; ++row) {
        for (int col = 0; col < shown; ++col) {
            std::printf("%9.3f ", matrix[row * size + col]);
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv)
{
    int size = 1024;
    int repeats = 1;
    if (argc > 3 || (argc >= 2 && !parsePositive(argv[1], 4096, size)) ||
        (argc == 3 && !parsePositive(argv[2], 100, repeats))) {
        std::fprintf(stderr, "Usage: a6.exe [matrix_size: 1..4096] [repeats: 1..100]\n");
        return 1;
    }
    const std::size_t count = static_cast<std::size_t>(size) * size;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> a(count);
    std::vector<float> b(count);
    std::vector<float> c(count);
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            // 小さな整数/8なので、CPUの検算でも正確に表せる。
            a[row * size + col] = ((row * 3 + col * 7) % 17 - 8) / 8.0f;
            b[row * size + col] = ((row * 5 + col * 11) % 19 - 9) / 8.0f;
        }
    }

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 threads(tileSize, tileSize);
    const dim3 blocks((size + tileSize - 1) / tileSize,
                       (size + tileSize - 1) / tileSize);
    // 初回の準備を済ませてから計測する。
    multiplyMatrices<<<blocks, threads>>>(deviceA, deviceB, deviceC, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("GPU: %s\n", properties.name);
    std::printf("C = A x B: %d x %d matrices, %d timed run(s)\n", size, size, repeats);
    std::fflush(stdout);
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        multiplyMatrices<<<blocks, threads>>>(deviceA, deviceB, deviceC, size);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float totalKernelMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalKernelMs, start, stop));
    CUDA_CHECK(cudaMemcpy(c.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceC));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));

    // 小さい行列は全要素、大きい行列は64要素をCPUで独立に検算する。
    const bool verifyAll = size <= 128;
    const std::size_t checks = verifyAll ? count : 64;
    for (std::size_t check = 0; check < checks; ++check) {
        const std::size_t index = verifyAll ? check : check * (count - 1) / (checks - 1);
        const std::size_t row = index / size;
        const std::size_t col = index % size;
        double expected = 0;
        for (int k = 0; k < size; ++k) {
            expected += static_cast<double>(a[row * size + k]) * b[k * size + col];
        }
        if (!std::isfinite(c[index]) || std::fabs(c[index] - expected) > 1e-5) {
            std::fprintf(stderr, "Mismatch at [%zu, %zu]: GPU=%.9f CPU=%.9f\n",
                         row, col, c[index], expected);
            return 1;
        }
    }
    const double averageMs = totalKernelMs / repeats;
    const double operations = 2.0 * size * size * size;
    std::printf("GPU kernel average: %.3f ms (%.1f GFLOP/s)\n",
                averageMs, operations / (averageMs * 1e6));
    std::printf("GPU transfers + all timed runs: %.3f ms\n", elapsedMs);
    std::printf("CPU verification: OK (%zu %s elements)\n", checks,
                verifyAll ? "of all" : "sampled");
    printCorner("A", a, size);
    printCorner("B", b, size);
    printCorner("C", c, size);
    return 0;
}
