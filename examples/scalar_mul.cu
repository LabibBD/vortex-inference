#include "vortex/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

__global__ void scalar_multiply(
    const float* input,
    float* output,
    float scalar,
    int n
) {
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = scalar * input[i];
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int THREADS = 256;
    constexpr float SCALAR = 2.5f;

    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    for (int i = 0; i < N; ++i) {
        h_input[i] =
            static_cast<float>(i) * 0.1f;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const int blocks =
        (N + THREADS - 1) / THREADS;

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    scalar_multiply<<<blocks, THREADS>>>(
        d_input,
        d_output,
        SCALAR,
        N
    );

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaEventRecord(start));

    scalar_multiply<<<blocks, THREADS>>>(
        d_input,
        d_output,
        SCALAR,
        N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernel_ms,
        start,
        stop
    ));

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_error = 0.0f;

    for (int i = 0; i < N; ++i) {
        const float expected =
            SCALAR * h_input[i];

        const float error =
            std::fabs(
                h_output[i] - expected
            );

        max_error =
            std::max(max_error, error);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "=================================\n";
    std::cout << " VORTEX Scalar Multiplication\n";
    std::cout << "=================================\n\n";

    std::cout << "Elements:       "
              << N << '\n';

    std::cout << "Scalar:         "
              << SCALAR << '\n';

    std::cout << "Threads/block:  "
              << THREADS << '\n';

    std::cout << "Blocks:         "
              << blocks << '\n';

    std::cout << std::fixed
              << std::setprecision(6);

    std::cout << "Kernel latency: "
              << kernel_ms
              << " ms\n";

    std::cout << "Maximum error:  "
              << max_error
              << '\n';

    if (max_error <= 1e-5f) {
        std::cout
            << "\n[VORTEX] Scalar multiplication PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Scalar multiplication FAILED\n";

    return 1;
}
