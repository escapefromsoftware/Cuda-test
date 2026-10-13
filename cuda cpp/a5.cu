#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

std::atomic<bool> stopRequested{false};

BOOL WINAPI requestStop(DWORD event)
{
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) {
        stopRequested.store(true, std::memory_order_relaxed);
        return TRUE;
    }
    return FALSE;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
            return 1;                                                           \
        }                                                                       \
    } while (false)

__global__ void manymanykeisan(float* output, int count, int rounds)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    float x = static_cast<float>(index & 1023) / 1024.0f;
    float y = 0.75f;
    // 各スレッドで反復計算。結果を保存し、計算が消されるのを防ぐ。
    #pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        x = fmaf(x, 0.9999f, 0.0001f);
        y = fmaf(y, 0.9998f, x * 0.0002f);
    }
    output[index] = x + y;
}

int main(int argc, char** argv)
{
    if (argc > 2) {
        std::fprintf(stderr, "Usage: a5.exe [seconds: 1..3600]\n");
        return 1;
    }
    long seconds = 10;
    if (argc == 2) {
        char* end = nullptr;
        errno = 0;
        seconds = std::strtol(argv[1], &end, 10);
        if (errno != 0 || *argv[1] == '\0' || *end != '\0' ||
            seconds < 1 || seconds > 3600) {
            std::fprintf(stderr, "seconds must be 1..3600\n");
            return 1;
        }
    }
    SetConsoleCtrlHandler(nullptr, FALSE);
    if (!SetConsoleCtrlHandler(requestStop, TRUE)) {
        std::fprintf(stderr, "Could not register Ctrl+C handler\n");
        return 1;
    }

    // GPU待ちでCPUを空回りさせず、GPUの計算に負荷をかける。
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int count = 1 << 20;
    constexpr int rounds = 4096;
    constexpr int threads = 256;
    constexpr int blocks = (count + threads - 1) / threads;
    constexpr std::size_t bytes = count * sizeof(float);
    float* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 未計算の要素を見つけられるよう、最初はNaNにする。
    CUDA_CHECK(cudaMemset(deviceOutput, 0xff, bytes));

    std::printf("GPU: %s\n", properties.name);
    std::printf("Load for %ld seconds; %d elements x %d rounds per launch.\n",
                seconds, count, rounds);
    std::printf("Press Ctrl+C to stop.\n");
    std::fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0;
    int nextReport = 1;
    unsigned long long launches = 0;
    while (!stopRequested.load(std::memory_order_relaxed) && elapsed < seconds) {
        manymanykeisan<<<blocks, threads>>>(deviceOutput, count, rounds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        ++launches;
        elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= nextReport) {
            std::printf("%.1f seconds: %llu launches completed\n", elapsed, launches);
            std::fflush(stdout);
            nextReport = static_cast<int>(elapsed) + 1;
        }
    }

    std::vector<float> output(count);
    CUDA_CHECK(cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceOutput));
    if (launches == 0) {
        std::printf("Stopped before launching a kernel.\n");
        return 0;
    }
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(output[index]) || output[index] < 0 || output[index] > 2) {
            std::fprintf(stderr, "Unexpected result at %d: %f\n", index, output[index]);
            return 1;
        }
    }
    const double billionRounds = static_cast<double>(count) * rounds * launches / 1e9;
    std::printf("%s after %.2f seconds: %.2f billion element-rounds.\n",
                stopRequested.load(std::memory_order_relaxed) ? "Stopped" : "Completed",
                elapsed, billionRounds);
    std::printf("All %d outputs are finite and in range. Sample: %.6f\n",
                count, output[0]);
    return 0;
}
