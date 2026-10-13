#include "rsa_common.cuh"

#include <cstring>

// Private exponent for p=1009, q=1013. Keep the private key in this program.
constexpr std::uint64_t rsaPrivateExponent = 832193;

__global__ void decryptMany(const std::uint32_t* ciphertext,
                            std::uint32_t* plaintext, std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        // Decryption uses the private key (n, d).
        plaintext[index] = powerMod(ciphertext[index], rsaPrivateExponent);
    }
}

int main(int argc, char** argv)
{
    if (argc > 3) {
        std::fprintf(stderr, "Usage: rsa_decrypt.exe [ciphertext_file] [plaintext_file]\n");
        return 1;
    }
    const char* inputPath = argc >= 2 ? argv[1] : "rsa_ciphertext.bin";
    const char* outputPath = argc == 3 ? argv[2] : "rsa_plaintext.bin";
    std::vector<std::uint32_t> ciphertext;
    if (!readCiphertext(inputPath, ciphertext)) {
        return 1;
    }
    char* fullInputPath = _fullpath(nullptr, inputPath, 0);
    char* fullOutputPath = _fullpath(nullptr, outputPath, 0);
    if (fullInputPath == nullptr || fullOutputPath == nullptr) {
        std::free(fullInputPath);
        std::free(fullOutputPath);
        std::fprintf(stderr, "Could not resolve input or output path\n");
        return 1;
    }
    const bool samePath = _stricmp(fullInputPath, fullOutputPath) == 0;
    std::free(fullInputPath);
    std::free(fullOutputPath);
    if (samePath) {
        std::fprintf(stderr, "Input and output paths must be different\n");
        return 1;
    }
    const auto count = ciphertext.size();
    std::vector<std::uint32_t> plaintext(count);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    GpuTiming timing;
    if (runGpu(ciphertext, plaintext, [&](auto* input, auto* output) {
            decryptMany<<<blocks, threads>>>(input, output, count);
        }, timing) != 0) {
        return 1;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (plaintext[index] != powerMod(ciphertext[index], rsaPrivateExponent) ||
            plaintext[index] != demoMessage(index)) {
            std::fprintf(stderr, "Decryption mismatch at %zu\n", index);
            return 1;
        }
    }
    if (!writeRsaFile(outputPath, rsaPlaintextMagic, plaintext)) {
        return 1;
    }
    std::printf("Decrypted %zu values with private key n=%u, d=%llu\n",
                count, rsaModulus, rsaPrivateExponent);
    std::printf("GPU kernel: %.3f ms; transfers + kernel: %.3f ms\n",
                timing.kernelMs, timing.transfersAndKernelMs);
    std::printf("Verification: OK. Plaintext saved to %s\n", outputPath);
    return 0;
}
