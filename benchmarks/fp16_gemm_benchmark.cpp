#include "vortex/benchmark.hpp"
#include "vortex/operators/cast.hpp"
#include "vortex/operators/matmul.hpp"
#include "vortex/tensor.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void benchmark_size(
    std::size_t size
) {
    using namespace vortex;

    const std::size_t elements =
        size * size;

    std::vector<float> a(
        elements,
        0.01f
    );

    std::vector<float> b(
        elements,
        0.02f
    );

    Tensor a32(
        {size, size},
        DataType::Float32,
        Device::CUDA
    );

    Tensor b32(
        {size, size},
        DataType::Float32,
        Device::CUDA
    );

    Tensor c32(
        {size, size},
        DataType::Float32,
        Device::CUDA
    );

    Tensor a16(
        {size, size},
        DataType::Float16,
        Device::CUDA
    );

    Tensor b16(
        {size, size},
        DataType::Float16,
        Device::CUDA
    );

    Tensor c16(
        {size, size},
        DataType::Float16,
        Device::CUDA
    );

    a32.copy_from_host(
        a.data(),
        elements * sizeof(float)
    );

    b32.copy_from_host(
        b.data(),
        elements * sizeof(float)
    );

    // Conversion is intentionally outside timed region.
    ops::float32_to_float16(
        a32,
        a16
    );

    ops::float32_to_float16(
        b32,
        b16
    );

    const BenchmarkStats fp32 =
        benchmark_cuda(
            [&]() {
                ops::matmul_cublas(
                    a32,
                    b32,
                    c32
                );
            },
            20,
            100
        );

    const BenchmarkStats fp16 =
        benchmark_cuda(
            [&]() {
                ops::matmul_cublas_fp16(
                    a16,
                    b16,
                    c16
                );
            },
            20,
            100
        );

    const double operations =
        2.0 *
        static_cast<double>(size) *
        static_cast<double>(size) *
        static_cast<double>(size);

    const double fp32_tflops =
        operations /
        (fp32.mean_ms * 1e-3) /
        1e12;

    const double fp16_tflops =
        operations /
        (fp16.mean_ms * 1e-3) /
        1e12;

    const double speedup =
        fp32.mean_ms /
        fp16.mean_ms;

    const double fp32_memory_mb =
        3.0 *
        static_cast<double>(
            elements * sizeof(float)
        ) /
        (1024.0 * 1024.0);

    const double fp16_memory_mb =
        3.0 *
        static_cast<double>(
            elements * 2
        ) /
        (1024.0 * 1024.0);

    std::cout
        << "\nMatrix: "
        << size << " x " << size
        << '\n';

    std::cout
        << "FP32 cuBLAS\n"
        << "  Mean:    "
        << fp32.mean_ms
        << " ms\n"
        << "  TFLOP/s: "
        << fp32_tflops
        << '\n';

    std::cout
        << "FP16 Tensor Core\n"
        << "  Mean:    "
        << fp16.mean_ms
        << " ms\n"
        << "  TFLOP/s: "
        << fp16_tflops
        << '\n';

    std::cout
        << "Speedup:   "
        << speedup
        << "x\n";

    std::cout
        << "Matrix memory (A+B+C):\n"
        << "  FP32: "
        << fp32_memory_mb
        << " MB\n"
        << "  FP16: "
        << fp16_memory_mb
        << " MB\n";
}

} // namespace

int main() {
    std::cout
        << "=================================\n"
        << " VORTEX FP16 GEMM Benchmark\n"
        << "=================================\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    benchmark_size(256);
    benchmark_size(512);
    benchmark_size(1024);
    benchmark_size(2048);

    return 0;
}
