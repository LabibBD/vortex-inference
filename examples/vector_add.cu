#include "vortex/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int THREADS = 256;

    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_a),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_b),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_c),
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const int blocks =
        (N + THREADS - 1) / THREADS;

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up kernel.
    vector_add<<<blocks, THREADS>>>(
        d_a,
        d_b,
        d_c,
        N
    );

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaEventRecord(start));

    vector_add<<<blocks, THREADS>>>(
        d_a,
        d_b,
        d_c,
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
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_error = 0.0f;

    for (int i = 0; i < N; ++i) {
        const float expected =
            h_a[i] + h_b[i];

        const float error =
            std::fabs(h_c[i] - expected);

        max_error =
            std::max(max_error, error);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::cout << "=================================\n";
    std::cout << " VORTEX Vector Addition\n";
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
              << kernel_ms
              << " ms\n";

    std::cout << "Maximum error:  "
              << max_error
              << '\n';

    if (max_error <= 1e-5f) {
        std::cout
            << "\n[VORTEX] Vector addition PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Vector addition FAILED\n";

    return 1;
}
