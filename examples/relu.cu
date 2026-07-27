#include "vortex/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

__global__ void relu_kernel(
    const float* input,
    float* output,
    int n
) {
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i] > 0.0f
            ? input[i]
            : 0.0f;
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int THREADS = 256;

    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    // Produces both positive and negative values.
    for (int i = 0; i < N; ++i) {
        h_input[i] =
            static_cast<float>((i % 201) - 100);
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
    relu_kernel<<<blocks, THREADS>>>(
        d_input,
        d_output,
        N
    );

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaEventRecord(start));

    relu_kernel<<<blocks, THREADS>>>(
        d_input,
        d_output,
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
            std::max(0.0f, h_input[i]);

        const float error =
            std::fabs(h_output[i] - expected);

        max_error =
            std::max(max_error, error);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "=================================\n";
    std::cout << " VORTEX ReLU\n";
    std::cout << "=================================\n\n";

    std::cout << "Elements:       "
              << N << '\n';

    std::cout << "Threads/block:  "
              << THREADS << '\n';

    std::cout << "Blocks:         "
              << blocks << '\n';

    std::cout << std::fixed
              << std::setprecision(6);

    std::cout << "Kernel latency: "
              << kernel_ms << " ms\n";

    std::cout << "Maximum error:  "
              << max_error << '\n';

    if (max_error <= 1e-5f) {
        std::cout
            << "\n[VORTEX] ReLU PASSED\n";
        return 0;
    }

    std::cerr
        << "\n[VORTEX] ReLU FAILED\n";
    return 1;
}
