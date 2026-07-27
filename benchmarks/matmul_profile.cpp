#include "vortex/operators/matmul.hpp"
#include "vortex/tensor.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t SIZE = 1024;
    constexpr std::size_t ELEMENTS = SIZE * SIZE;

    std::vector<float> a(ELEMENTS, 0.1f);
    std::vector<float> b(ELEMENTS, 0.2f);

    Tensor d_a(
        {SIZE, SIZE},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_b(
        {SIZE, SIZE},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_c(
        {SIZE, SIZE},
        DataType::Float32,
        Device::CUDA
    );

    d_a.copy_from_host(
        a.data(),
        ELEMENTS * sizeof(float)
    );

    d_b.copy_from_host(
        b.data(),
        ELEMENTS * sizeof(float)
    );

    // Warm-up
    ops::matmul_naive(d_a, d_b, d_c);
    ops::matmul_tiled(d_a, d_b, d_c);

    std::cout << "[VORTEX] Profiling naive GEMM\n";
    ops::matmul_naive(d_a, d_b, d_c);

    std::cout << "[VORTEX] Profiling tiled GEMM\n";
    ops::matmul_tiled(d_a, d_b, d_c);

    std::cout << "[VORTEX] Profiling complete\n";

    return 0;
}
