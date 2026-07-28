#include "vortex/benchmark.hpp"
#include "vortex/memory_pool.hpp"
#include "vortex/numerics.hpp"
#include "vortex/operators/int8_linear.hpp"
#include "vortex/operators/linear.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct HostQuantizedWeights {
    std::vector<std::int8_t> values;
    std::vector<float> scales;
};

struct CaseResult {
    std::size_t rows{};
    std::size_t input_features{};
    std::size_t output_features{};

    double fp32_mean_ms{};
    double fp32_median_ms{};
    double fp32_p95_ms{};

    double int8_mean_ms{};
    double int8_median_ms{};
    double int8_p95_ms{};

    double speedup{};

    float maximum_error{};
    double mean_error{};

    std::size_t fp32_weight_bytes{};
    std::size_t int8_weight_bytes{};
    std::size_t workspace_bytes{};

    double weight_compression{};
};

int quantize_value(
    float value,
    float scale
) {
    const int rounded =
        static_cast<int>(
            std::nearbyint(
                value / scale
            )
        );

    return std::max(
        -127,
        std::min(
            127,
            rounded
        )
    );
}

HostQuantizedWeights
quantize_weights(
    const std::vector<float>& weight,
    std::size_t input_features,
    std::size_t output_features
) {
    if (
        weight.size() !=
        input_features *
        output_features
    ) {
        throw std::runtime_error(
            "Weight size mismatch"
        );
    }

    HostQuantizedWeights result;

    result.values.resize(
        weight.size()
    );

    result.scales.resize(
        output_features
    );

    for (
        std::size_t column = 0;
        column < output_features;
        ++column
    ) {
        float maximum =
            0.0f;

        for (
            std::size_t row = 0;
            row < input_features;
            ++row
        ) {
            maximum =
                std::max(
                    maximum,
                    std::fabs(
                        weight[
                            row *
                            output_features +
                            column
                        ]
                    )
                );
        }

        const float scale =
            maximum > 0.0f
                ? maximum / 127.0f
                : 1.0f;

        result.scales[column] =
            scale;

        for (
            std::size_t row = 0;
            row < input_features;
            ++row
        ) {
            const std::size_t index =
                row *
                output_features +
                column;

            result.values[index] =
                static_cast<std::int8_t>(
                    quantize_value(
                        weight[index],
                        scale
                    )
                );
        }
    }

    return result;
}

CaseResult run_case(
    std::size_t rows,
    std::size_t input_features,
    std::size_t output_features,
    int iterations,
    unsigned int seed
) {
    using namespace vortex;

    constexpr int WARMUP =
        30;

    std::mt19937 generator(
        seed
    );

    std::normal_distribution<float>
        input_distribution(
            0.0f,
            1.0f
        );

    std::normal_distribution<float>
        weight_distribution(
            0.0f,
            0.20f
        );

    std::normal_distribution<float>
        bias_distribution(
            0.0f,
            0.05f
        );

    std::vector<float> host_input(
        rows *
        input_features
    );

    std::vector<float> host_weight(
        input_features *
        output_features
    );

    std::vector<float> host_bias(
        output_features
    );

    for (float& value : host_input) {
        value =
            input_distribution(
                generator
            );
    }

    for (float& value : host_weight) {
        value =
            weight_distribution(
                generator
            );
    }

    for (float& value : host_bias) {
        value =
            bias_distribution(
                generator
            );
    }

    const HostQuantizedWeights
        quantized =
            quantize_weights(
                host_weight,
                input_features,
                output_features
            );

    /*
     * Linear3D requires rank-3 input/output.
     *
     * [1, rows, features] is equivalent to flattening
     * batch and token dimensions into rows.
     */
    Tensor input(
        {
            1,
            rows,
            input_features
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor fp32_weight(
        {
            input_features,
            output_features
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor bias(
        {output_features},
        DataType::Float32,
        Device::CUDA
    );

    Tensor fp32_output(
        {
            1,
            rows,
            output_features
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor int8_output(
        {
            1,
            rows,
            output_features
        },
        DataType::Float32,
        Device::CUDA
    );

    input.copy_from_host(
        host_input.data(),
        host_input.size() *
            sizeof(float)
    );

    fp32_weight.copy_from_host(
        host_weight.data(),
        host_weight.size() *
            sizeof(float)
    );

    bias.copy_from_host(
        host_bias.data(),
        host_bias.size() *
            sizeof(float)
    );

    ops::Int8LinearWeights
        int8_weights(
            input_features,
            output_features
        );

    int8_weights.copy_from_host(
        quantized.values.data(),
        quantized.scales.data()
    );

    ops::Int8LinearWorkspace
        workspace(
            rows,
            input_features
        );

    // ==================================================
    // Initial execution and numerical comparison
    // ==================================================

    ops::linear_3d(
        input,
        fp32_weight,
        bias,
        fp32_output
    );

    ops::linear_3d_int8_dynamic(
        input,
        int8_weights,
        bias,
        workspace,
        int8_output
    );

    if (
        !numerics::all_finite(
            fp32_output
        ) ||
        !numerics::all_finite(
            int8_output
        )
    ) {
        throw std::runtime_error(
            "Benchmark produced NaN or Inf"
        );
    }

    const std::size_t output_elements =
        rows *
        output_features;

    std::vector<float> fp32_host(
        output_elements
    );

    std::vector<float> int8_host(
        output_elements
    );

    fp32_output.copy_to_host(
        fp32_host.data(),
        fp32_host.size() *
            sizeof(float)
    );

    int8_output.copy_to_host(
        int8_host.data(),
        int8_host.size() *
            sizeof(float)
    );

    float maximum_error =
        0.0f;

    double mean_error =
        0.0;

    for (
        std::size_t index = 0;
        index < output_elements;
        ++index
    ) {
        const float error =
            std::fabs(
                int8_host[index] -
                fp32_host[index]
            );

        maximum_error =
            std::max(
                maximum_error,
                error
            );

        mean_error +=
            error;
    }

    mean_error /=
        static_cast<double>(
            output_elements
        );

    // ==================================================
    // Performance measurement
    // ==================================================

    const BenchmarkStats fp32 =
        benchmark_cuda(
            [&]() {
                ops::linear_3d(
                    input,
                    fp32_weight,
                    bias,
                    fp32_output
                );
            },
            WARMUP,
            iterations
        );

    const BenchmarkStats int8 =
        benchmark_cuda(
            [&]() {
                ops::linear_3d_int8_dynamic(
                    input,
                    int8_weights,
                    bias,
                    workspace,
                    int8_output
                );
            },
            WARMUP,
            iterations
        );

    const std::size_t
        fp32_weight_bytes =
            input_features *
            output_features *
            sizeof(float);

    const std::size_t
        int8_weight_bytes =
            input_features *
            output_features *
            sizeof(std::int8_t) +
            output_features *
            sizeof(float);

    const std::size_t
        workspace_bytes =
            rows *
            input_features *
            sizeof(std::int8_t) +
            rows *
            sizeof(float);

    return CaseResult{
        rows,
        input_features,
        output_features,

        fp32.mean_ms,
        fp32.median_ms,
        fp32.p95_ms,

        int8.mean_ms,
        int8.median_ms,
        int8.p95_ms,

        fp32.mean_ms /
            int8.mean_ms,

        maximum_error,
        mean_error,

        fp32_weight_bytes,
        int8_weight_bytes,
        workspace_bytes,

        static_cast<double>(
            fp32_weight_bytes
        ) /
        static_cast<double>(
            int8_weight_bytes
        )
    };
}

void print_case(
    const CaseResult& result
) {
    std::cout
        << "\nShape: "
        << result.rows
        << " x "
        << result.input_features
        << " x "
        << result.output_features
        << '\n';

    std::cout
        << "FP32 mean:             "
        << result.fp32_mean_ms
        << " ms\n";

    std::cout
        << "FP32 median:           "
        << result.fp32_median_ms
        << " ms\n";

    std::cout
        << "FP32 P95:              "
        << result.fp32_p95_ms
        << " ms\n";

    std::cout
        << "INT8 mean:             "
        << result.int8_mean_ms
        << " ms\n";

    std::cout
        << "INT8 median:           "
        << result.int8_median_ms
        << " ms\n";

    std::cout
        << "INT8 P95:              "
        << result.int8_p95_ms
        << " ms\n";

    std::cout
        << "INT8 speedup:          "
        << result.speedup
        << "x\n";

    std::cout
        << "Maximum FP32 error:    "
        << result.maximum_error
        << '\n';

    std::cout
        << "Mean FP32 error:       "
        << result.mean_error
        << '\n';

    std::cout
        << "FP32 weight storage:   "
        << result.fp32_weight_bytes
        << " bytes\n";

    std::cout
        << "INT8 weight storage:   "
        << result.int8_weight_bytes
        << " bytes\n";

    std::cout
        << "Weight compression:    "
        << result.weight_compression
        << "x\n";

    std::cout
        << "INT8 workspace:        "
        << result.workspace_bytes
        << " bytes\n";
}

} // namespace

int main() {
    using namespace vortex;

    try {
        MemoryPool::global().clear();

        std::filesystem::
            create_directories(
                "../results"
            );

        std::ofstream csv(
            "../results/"
            "phase26_int8_benchmark.csv"
        );

        if (!csv) {
            throw std::runtime_error(
                "Could not create INT8 benchmark CSV"
            );
        }

        csv
            << "rows,input_features,output_features,"
            << "fp32_mean_ms,fp32_median_ms,fp32_p95_ms,"
            << "int8_mean_ms,int8_median_ms,int8_p95_ms,"
            << "int8_speedup,max_error,mean_error,"
            << "fp32_weight_bytes,int8_weight_bytes,"
            << "weight_compression,int8_workspace_bytes\n";

        std::cout
            << "=================================\n"
            << " VORTEX FP32 vs Dynamic INT8\n"
            << "=================================\n";

        std::cout
            << std::fixed
            << std::setprecision(6);

        const std::vector<CaseResult>
            cases{
                run_case(
                    64,
                    32,
                    32,
                    500,
                    2611
                ),

                run_case(
                    64,
                    32,
                    64,
                    500,
                    2612
                ),

                run_case(
                    64,
                    64,
                    32,
                    500,
                    2613
                ),

                run_case(
                    256,
                    128,
                    128,
                    250,
                    2614
                )
            };

        for (
            const CaseResult& result :
            cases
        ) {
            print_case(
                result
            );

            csv
                << result.rows
                << ','
                << result.input_features
                << ','
                << result.output_features
                << ','
                << result.fp32_mean_ms
                << ','
                << result.fp32_median_ms
                << ','
                << result.fp32_p95_ms
                << ','
                << result.int8_mean_ms
                << ','
                << result.int8_median_ms
                << ','
                << result.int8_p95_ms
                << ','
                << result.speedup
                << ','
                << result.maximum_error
                << ','
                << result.mean_error
                << ','
                << result.fp32_weight_bytes
                << ','
                << result.int8_weight_bytes
                << ','
                << result.weight_compression
                << ','
                << result.workspace_bytes
                << '\n';
        }

        std::cout
            << "\n[VORTEX] Saved "
            << "results/phase26_int8_benchmark.csv\n";

        std::cout
            << "[VORTEX] INT8 benchmark PASSED\n";

        return 0;
    }
    catch (
        const std::exception& error
    ) {
        std::cerr
            << "[VORTEX][ERROR] "
            << error.what()
            << '\n';

        return 1;
    }
}
