#include "vortex/operators/cast.hpp"
#include "vortex/operators/matmul.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t M = 128;
    constexpr std::size_t K = 128;
    constexpr std::size_t N = 128;

    std::vector<float> a(M * K);
    std::vector<float> b(K * N);

    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.02f;
    }

    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] =
            static_cast<float>(
                static_cast<int>(i % 13) - 6
            ) * 0.02f;
    }

    // CPU FP32 reference.
    std::vector<float> reference(
        M * N,
        0.0f
    );

    for (std::size_t row = 0; row < M; ++row) {
        for (std::size_t col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                sum +=
                    a[row * K + k] *
                    b[k * N + col];
            }

            reference[row * N + col] =
                sum;
        }
    }

    // FP32 CUDA tensors.
    Tensor a32(
        {M, K},
        DataType::Float32,
        Device::CUDA
    );

    Tensor b32(
        {K, N},
        DataType::Float32,
        Device::CUDA
    );

    a32.copy_from_host(
        a.data(),
        a.size() * sizeof(float)
    );

    b32.copy_from_host(
        b.data(),
        b.size() * sizeof(float)
    );

    // FP16 tensors.
    Tensor a16(
        {M, K},
        DataType::Float16,
        Device::CUDA
    );

    Tensor b16(
        {K, N},
        DataType::Float16,
        Device::CUDA
    );

    Tensor c16(
        {M, N},
        DataType::Float16,
        Device::CUDA
    );

    Tensor c32(
        {M, N},
        DataType::Float32,
        Device::CUDA
    );

    ops::float32_to_float16(
        a32,
        a16
    );

    ops::float32_to_float16(
        b32,
        b16
    );

    ops::matmul_cublas_fp16(
        a16,
        b16,
        c16
    );

    ops::float16_to_float32(
        c16,
        c32
    );

    std::vector<float> result(M * N);

    c32.copy_to_host(
        result.data(),
        result.size() * sizeof(float)
    );

    float max_error = 0.0f;
    double mean_error = 0.0;

    for (std::size_t i = 0; i < result.size(); ++i) {
        const float error =
            std::fabs(
                result[i] -
                reference[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );

        mean_error += error;
    }

    mean_error /=
        static_cast<double>(
            result.size()
        );

    std::cout
        << "=================================\n"
        << " VORTEX FP16 GEMM Test\n"
        << "=================================\n\n";

    std::cout
        << "Matrix:            "
        << M << " x " << K
        << " x " << N << '\n';

    std::cout
        << "Maximum abs error: "
        << max_error << '\n';

    std::cout
        << "Mean abs error:    "
        << mean_error << '\n';

    if (
        max_error <= 0.05f &&
        mean_error <= 0.01
    ) {
        std::cout
            << "\n[VORTEX] FP16 GEMM test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] FP16 GEMM test FAILED\n";

    return 1;
}
