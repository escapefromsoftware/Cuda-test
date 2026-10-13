#include "rsa_common.cuh"

__global__ void encryptMany(const std::uint32_t* plaintext,
                            std::uint32_t* ciphertext, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        // Encryption uses the public key (n, e).
        ciphertext[index] = powerMod(plaintext[index], rsaPublicExponent);
    }
}

int main(int argc, char** argv)
{
    if (argc > 3) {
        std::fprintf(stderr, "Usage: rsa_encrypt.exe [elements] [ciphertext_file]\n");
        return 1;
    }
    std::size_t count = 1u << 20;
    if (argc >= 2 && !parseCount(argv[1], count)) {
        return 1;
    }
    const char* outputPath = argc == 3 ? argv[2] : "rsa_ciphertext.bin";
    std::vector<std::uint32_t> plaintext(count);
    std::vector<std::uint32_t> ciphertext(count);
    for (std::size_t index = 0; index < count; ++index) {
        plaintext[index] = demoMessage(index);
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    GpuTiming timing;
    if (runGpu(plaintext, ciphertext, [&](auto* input, auto* output) {
            encryptMany<<<blocks, threads>>>(input, output, count);
        }, timing) != 0) {
        return 1;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (ciphertext[index] != powerMod(plaintext[index], rsaPublicExponent)) {
            std::fprintf(stderr, "Encryption mismatch at %zu\n", index);
            return 1;
        }
    }
    if (!writeRsaFile(outputPath, rsaCiphertextMagic, ciphertext)) {
        return 1;
    }
    std::printf("Encrypted %zu values with public key n=%u, e=%u\n",
                count, rsaModulus, rsaPublicExponent);
    std::printf("GPU kernel: %.3f ms; transfers + kernel: %.3f ms\n",
                timing.kernelMs, timing.transfersAndKernelMs);
    std::printf("Verification: OK. Ciphertext saved to %s\n", outputPath);
    return 0;
}
