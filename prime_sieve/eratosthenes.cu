#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>

constexpr int limit = 1000;
constexpr int threadsPerBlock = 256;
constexpr int phaseCount = 30;

// GPUの各段階の記録。個々のスレッド内の実行順序は記録しない。
struct PhaseTrace {
    int p;
    int selected;
    int marks;
    int newlyRemoved;
    int remaining;
    unsigned long long beginCycles;
    unsigned long long endCycles;
};

struct SieveCounts {
    int candidates;
    int marks;
    int newlyRemoved;
    int maxMarksPerThreadPerPrime;
    int marksByPrime[32];
};

struct DeviceMemory {
    int* flags = nullptr;
    SieveCounts* counts = nullptr;
    PhaseTrace* trace = nullptr;
    ~DeviceMemory() {
        if (trace) cudaFree(trace);
        if (counts) cudaFree(counts);
        if (flags) cudaFree(flags);
    }
};

#define CUDA_CHECK(call)                                                   \
    do {                                                                  \
        const cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                        \
            std::cerr << #call << ": " << cudaGetErrorString(error) << '\n'; \
            return 1;                                                     \
        }                                                                 \
    } while (false)

// 1ブロック内で篩の表を共有する。素数ごとの処理には順序が必要。
// 同じ素数の倍数は、256スレッドが分担して並列に除外する。
__global__ void sieve(int* result, SieveCounts* counts, PhaseTrace* trace) {
    __shared__ int isPrime[limit + 1];
    __shared__ int primeSelected;
    __shared__ int removedBeforePhase;
    __shared__ unsigned long long initialCycle;
    const int thread = threadIdx.x;

    if (thread == 0) initialCycle = clock64();
    for (int number = thread; number <= limit; number += blockDim.x) {
        isPrime[number] = (number >= 2);
    }
    __syncthreads();

    for (int p = 2; p * p <= limit; ++p) {
        if (thread == 0) {
            primeSelected = isPrime[p];
            removedBeforePhase = counts->newlyRemoved;
            trace[p - 2].p = p;
            trace[p - 2].selected = primeSelected;
            trace[p - 2].beginCycles = clock64() - initialCycle;
            ++counts->candidates;
        }
        __syncthreads();
        if (primeSelected) {
            int marksForThread = 0;
            // p*p未満の倍数は、すでに小さい素数で除外済み。
            for (int multiple = p * p + thread * p; multiple <= limit;
                 multiple += blockDim.x * p) {
                const int previouslyPrime = atomicExch(&isPrime[multiple], 0);
                atomicAdd(&counts->marks, 1);
                atomicAdd(&counts->marksByPrime[p], 1);
                if (previouslyPrime) {
                    atomicAdd(&counts->newlyRemoved, 1);
                }
                ++marksForThread;
            }
            atomicMax(&counts->maxMarksPerThreadPerPrime, marksForThread);
        }

        // 全スレッドが倍数を除外してから、次の候補を判定する。
        // 素数でない場合も全スレッドがこの同期に参加する。
        __syncthreads();
        if (thread == 0) {
            // 同期後なので、この段階の全スレッドの結果が確定している。
            trace[p - 2].endCycles = clock64() - initialCycle;
            trace[p - 2].marks = primeSelected ? counts->marksByPrime[p] : 0;
            trace[p - 2].newlyRemoved = counts->newlyRemoved - removedBeforePhase;
            trace[p - 2].remaining = limit - 1 - counts->newlyRemoved;
        }
    }

    for (int number = thread; number <= limit; number += blockDim.x) {
        result[number] = isPrime[number];
    }
}

int main() {
    const auto programStart = std::chrono::steady_clock::now();
    int hostStep = 0;
    const auto logHost = [&](const char* message) {
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - programStart).count();
        std::cout << "[CPU step " << ++hostStep << ", +"
                  << std::fixed << std::setprecision(3) << milliseconds
                  << " ms] " << message << std::endl;
    };
    logHost("Start: prepare CUDA device");
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::cout << "GPU: " << properties.name << '\n';

    DeviceMemory gpu;
    logHost("Allocate GPU memory and reset operation counters");
    CUDA_CHECK(cudaMalloc(&gpu.flags, (limit + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.counts, sizeof(SieveCounts)));
    CUDA_CHECK(cudaMalloc(&gpu.trace, phaseCount * sizeof(PhaseTrace)));
    CUDA_CHECK(cudaMemset(gpu.counts, 0, sizeof(SieveCounts)));

    logHost("Launch sieve kernel: 1 block, 256 threads");
    sieve<<<1, threadsPerBlock>>>(gpu.flags, gpu.counts, gpu.trace);
    CUDA_CHECK(cudaGetLastError());
    logHost("Wait for GPU completion");
    CUDA_CHECK(cudaDeviceSynchronize());
    logHost("GPU finished; copy flags, counters and phase trace to CPU");

    std::array<int, limit + 1> isPrime{};
    SieveCounts counts{};
    std::array<PhaseTrace, phaseCount> trace{};
    CUDA_CHECK(cudaMemcpy(isPrime.data(), gpu.flags, sizeof(isPrime), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&counts, gpu.counts, sizeof(counts), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trace.data(), gpu.trace, sizeof(trace), cudaMemcpyDeviceToHost));

    logHost("Display recorded GPU timeline (execution has already finished)");
    std::cout << "\nGPU timeline: ordered phases, cycle offsets from thread 0 start\n"
              << "  Initialize shared table: 0 and 1 excluded; 999 candidates remain\n";
    int cumulativeRemoved = 0;
    unsigned long long previousEnd = 0;
    for (int phase = 0; phase < phaseCount; ++phase) {
        const PhaseTrace& entry = trace[phase];
        cumulativeRemoved += entry.newlyRemoved;
        if (entry.p != phase + 2 || entry.selected != isPrime[entry.p] ||
            entry.marks != counts.marksByPrime[entry.p] || entry.newlyRemoved < 0 ||
            entry.newlyRemoved > entry.marks || entry.remaining != limit - 1 - cumulativeRemoved ||
            entry.beginCycles < previousEnd || entry.endCycles < entry.beginCycles) {
            std::cerr << "GPU timeline verification failed at phase " << phase + 1 << '\n';
            return 1;
        }
        previousEnd = entry.endCycles;
        std::cout << "  [GPU phase " << std::setw(2) << phase + 1
                  << ", cycles " << entry.beginCycles << ".." << entry.endCycles
                  << "] p=" << entry.p;
        if (entry.selected) {
            std::cout << " PRIME -> mark multiples in parallel\n    Targets: ";
            const int sampleCount = entry.marks < 6 ? entry.marks : 6;
            for (int sample = 0; sample < sampleCount; ++sample) {
                if (sample) std::cout << ", ";
                std::cout << entry.p * entry.p + sample * entry.p;
            }
            if (entry.marks > sampleCount) {
                std::cout << ", ... , " << (limit / entry.p) * entry.p;
            }
            std::cout << " (numeric order; GPU threads run in parallel)\n"
                      << "    Marks=" << entry.marks
                      << ", newly removed=" << entry.newlyRemoved;
        } else {
            std::cout << " COMPOSITE -> skip; no multiples marked";
        }
        std::cout << "; remaining candidates=" << entry.remaining << '\n';
    }
    if (cumulativeRemoved != counts.newlyRemoved) {
        std::cerr << "GPU timeline total failed verification\n";
        return 1;
    }
    std::cout << "  Write final flags from shared memory to GPU output buffer\n\n";

    // 篩とは独立した試し割りで、0〜1000のGPU結果を全件検算する。
    logHost("Verify all GPU flags on CPU using independent trial division");
    int expectedMarks = 0;
    int primeCount = 0;
    int largestPrime = 0;
    for (int number = 0; number <= limit; ++number) {
        bool expected = (number >= 2);
        for (int divisor = 2; divisor * divisor <= number; ++divisor) {
            if (number % divisor == 0) {
                expected = false;
                break;
            }
        }
        if (isPrime[number] != static_cast<int>(expected)) {
            std::cerr << "GPU verification failed at " << number << '\n';
            return 1;
        }
        if (expected) {
            ++primeCount;
            largestPrime = number;
            if (number * number <= limit) {
                const int expectedForPrime = limit / number - number + 1;
                expectedMarks += expectedForPrime;
                if (counts.marksByPrime[number] != expectedForPrime) {
                    std::cerr << "Mark count mismatch for prime " << number << '\n';
                    return 1;
                }
            }
        }
    }
    const int expectedCandidates = 30;
    const int expectedMaxPerThread = ((limit / 2 - 1) + threadsPerBlock - 1) / threadsPerBlock;
    if (counts.candidates != expectedCandidates || counts.marks != expectedMarks ||
        counts.newlyRemoved != limit - 1 - primeCount ||
        counts.maxMarksPerThreadPerPrime != expectedMaxPerThread) {
        std::cerr << "GPU operation counts failed verification\n";
        return 1;
    }

    logHost("Verification passed; print prime list and operation summary");
    std::cout << "\nCUDA sieve: 1.." << limit
              << "\nKernel launches: 1 (1 block, " << threadsPerBlock << " threads)\n"
              << "\nPrime   First multiple   Marking operations\n";
    for (int p = 2; p * p <= limit; ++p) {
        if (isPrime[p]) {
            std::cout << std::setw(5) << p << std::setw(17) << p * p
                      << std::setw(21) << counts.marksByPrime[p] << '\n';
        }
    }

    int printed = 0;
    std::cout << "\nPrimes:\n";
    for (int number = 2; number <= limit; ++number) {
        if (isPrime[number]) {
            std::cout << std::setw(5) << number;
            if (++printed % 16 == 0) std::cout << '\n';
        }
    }
    std::cout << "\n\nPrime count: " << primeCount
              << "\nLargest prime: " << largestPrime
              << "\nCandidate checks / sequential sieve phases: " << counts.candidates
              << "\nMarking operations (all threads, including duplicates): " << counts.marks
              << "\nDistinct composite numbers removed: " << counts.newlyRemoved
              << "\nLogical sieve work (candidate checks + marks): " << counts.candidates + counts.marks
              << "\nMaximum marking-loop iterations per thread per prime: "
              << counts.maxMarksPerThreadPerPrime
              << "\nCPU verification: OK (all 1001 flags, GPU operation counts and phase trace)\n";
    logHost("Done; GPU buffers will be released");
}
