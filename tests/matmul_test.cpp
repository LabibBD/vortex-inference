#include "vortex/operators/matmul.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t M = 127;
    constexpr std::size_t K = 83;
    constexpr std::size_t N = 91;

    std::vector<float> a(M * K);
    std::vector<float> b(K * N);

    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.05f;
    }

    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] =
            static_cast<float>(
                static_cast<int>(i % 13) - 6
            ) * 0.04f;
    }

    std::vector<float> cpu(M * N, 0.0f);

    for (std::size_t row = 0; row < M; ++row) {
        for (std::size_t col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (std::size_t i = 0; i < K; ++i) {
                sum +=
                    a[row * K + i] *
                    b[i * N + col];
            }

            cpu[row * N + col] = sum;
        }
    }

    Tensor d_a(
        {M, K},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_b(
        {K, N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_naive(
        {M, N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_tiled(
        {M, N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_cublas(
        {M, N},
        DataType::Float32,
        Device::CUDA
    );

    d_a.copy_from_host(
        a.data(),
        a.size() * sizeof(float)
    );

    d_b.copy_from_host(
        b.data(),
        b.size() * sizeof(float)
    );

    ops::matmul_naive(
        d_a,
        d_b,
        d_naive
    );

    ops::matmul_tiled(
        d_a,
        d_b,
        d_tiled
    );

    ops::matmul_cublas(
        d_a,
        d_b,
        d_cublas
    );

    std::vector<float> naive(M * N);
    std::vector<float> tiled(M * N);
    std::vector<float> cublas(M * N);

    d_naive.copy_to_host(
        naive.data(),
        naive.size() * sizeof(float)
    );

    d_tiled.copy_to_host(
        tiled.data(),
        tiled.size() * sizeof(float)
    );

    d_cublas.copy_to_host(
        cublas.data(),
        cublas.size() * sizeof(float)
    );

    float naive_error = 0.0f;
    float tiled_error = 0.0f;
    float cublas_error = 0.0f;

    for (std::size_t i = 0; i < cpu.size(); ++i) {
        naive_error =
            std::max(
                naive_error,
                std::fabs(
                    naive[i] - cpu[i]
                )
            );

        tiled_error =
            std::max(
                tiled_error,
                std::fabs(
                    tiled[i] - cpu[i]
                )
            );

        cublas_error =
            std::max(
                cublas_error,
                std::fabs(
                    cublas[i] - cpu[i]
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX GEMM Correctness\n"
        << "=================================\n\n";

    std::cout
        << "Naive max error: "
        << naive_error
        << '\n';

    std::cout
        << "Tiled max error: "
        << tiled_error
        << '\n';

    std::cout
        << "cuBLAS max error:"
        << cublas_error
        << '\n';

    const bool passed =
        naive_error <= 1e-4f &&
        tiled_error <= 1e-4f &&
        cublas_error <= 1e-4f;

    if (passed) {
        std::cout
            << "\n[VORTEX] GEMM correctness PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] GEMM correctness FAILED\n";

    return 1;
}
