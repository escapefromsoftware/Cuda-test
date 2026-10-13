#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

// Public key for the small, unpadded RSA demonstration.
constexpr std::uint32_t rsaModulus = 1022117;
constexpr std::uint32_t rsaPublicExponent = 65537;
constexpr std::size_t rsaMaxCount = 1u << 24;
constexpr std::uint32_t rsaCiphertextMagic = 0x31415352; // "RSA1" in little endian
constexpr std::uint32_t rsaPlaintextMagic = 0x31505352; // "RSP1" in little endian

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(error));                            \
            return 1;                                                           \
        }                                                                       \
    } while (false)

// Products are safe in uint64_t only for this small demonstration modulus.
__host__ __device__ inline std::uint32_t powerMod(std::uint64_t base,
                                                 std::uint64_t exponent)
{
    std::uint64_t result = 1;
    while (exponent != 0) {
        if ((exponent & 1) != 0) {
            result = result * base % rsaModulus;
        }
        base = base * base % rsaModulus;
        exponent >>= 1;
    }
    return static_cast<std::uint32_t>(result);
}

inline std::uint32_t demoMessage(std::size_t index)
{
    return static_cast<std::uint32_t>((index * 17 + 42) % rsaModulus);
}

inline bool parseCount(const char* text, std::size_t& count)
{
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(text, &end, 10);
    if (*text == '-' || *text == '\0' || errno != 0 || *end != '\0' ||
        value < 1 || value > rsaMaxCount) {
        std::fprintf(stderr, "elements must be 1..16777216\n");
        return false;
    }
    count = static_cast<std::size_t>(value);
    return true;
}

// Windows little-endian format: four uint32_t header fields, then uint32_t values.
// Header: magic, public modulus, public exponent, element count. No private key.
inline bool writeRsaFile(const char* path, std::uint32_t magic,
                         const std::vector<std::uint32_t>& values)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::array<std::uint32_t, 4> header = {
        magic, rsaModulus, rsaPublicExponent,
        static_cast<std::uint32_t>(values.size())};
    if (!file) {
        std::fprintf(stderr, "Cannot open output: %s\n", path);
        return false;
    }
    file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    file.close();
    if (!file) {
        std::fprintf(stderr, "Could not write output: %s\n", path);
        return false;
    }
    return true;
}

inline bool readCiphertext(const char* path, std::vector<std::uint32_t>& values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "Cannot open ciphertext: %s (run rsa_encrypt first)\n", path);
        return false;
    }
    const auto length = file.tellg();
    file.seekg(0);
    std::array<std::uint32_t, 4> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!file || header[0] != rsaCiphertextMagic || header[1] != rsaModulus ||
        header[2] != rsaPublicExponent || header[3] < 1 || header[3] > rsaMaxCount ||
        length != static_cast<std::streamoff>(sizeof(header) +
                                              header[3] * sizeof(std::uint32_t))) {
        std::fprintf(stderr, "Invalid or unsupported ciphertext file: %s\n", path);
        return false;
    }
    values.resize(header[3]);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    if (!file) {
        std::fprintf(stderr, "Could not read ciphertext: %s\n", path);
        return false;
    }
    for (const auto value : values) {
        if (value >= rsaModulus) {
            std::fprintf(stderr, "Ciphertext value is outside the modulus\n");
            return false;
        }
    }
    return true;
}

struct GpuTiming {
    float kernelMs = 0;
    double transfersAndKernelMs = 0;
};

// Both programs use the same measurement setup; their kernels live separately.
template <typename Launch>
int runGpu(const std::vector<std::uint32_t>& input,
           std::vector<std::uint32_t>& output, Launch launch, GpuTiming& timing)
{
    const auto bytes = input.size() * sizeof(input[0]);
    std::uint32_t* deviceInput = nullptr;
    std::uint32_t* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    CUDA_CHECK(cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice));
    launch(deviceInput, deviceOutput);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    launch(deviceInput, deviceOutput);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timing.kernelMs, start, stop));
    CUDA_CHECK(cudaMemcpy(output.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));
    timing.transfersAndKernelMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceInput));
    return 0;
}
