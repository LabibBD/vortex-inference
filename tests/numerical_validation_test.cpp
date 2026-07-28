#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"
#include "vortex/numerics.hpp"
#include "vortex/operators/activations.hpp"
#include "vortex/operators/attention.hpp"
#include "vortex/operators/scaled_softmax.hpp"
#include "vortex/operators/softmax.hpp"
#include "vortex/tensor.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double GELU_FACTOR =
    0.7978845608028654;

constexpr double GELU_COEFFICIENT =
    0.044715;

double gelu_reference(
    double value
) {
    return 0.5 *
        value *
        (
            1.0 +
            std::tanh(
                GELU_FACTOR *
                (
                    value +
                    GELU_COEFFICIENT *
                    value *
                    value *
                    value
                )
            )
        );
}

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(
            message
        );
    }
}

struct ErrorMetrics {
    float maximum{};
    double mean{};
};

ErrorMetrics compare(
    const std::vector<float>& result,
    const std::vector<float>& reference
) {
    require(
        result.size() ==
            reference.size(),
        "Comparison size mismatch"
    );

    ErrorMetrics metrics;

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {
        const float error =
            std::fabs(
                result[index] -
                reference[index]
            );

        metrics.maximum =
            std::max(
                metrics.maximum,
                error
            );

        metrics.mean +=
            error;
    }

    if (!result.empty()) {
        metrics.mean /=
            static_cast<double>(
                result.size()
            );
    }

    return metrics;
}

// ============================================================
// Stable Softmax
// ============================================================

struct SoftmaxMetrics {
    float standard_error{};
    float scaled_error{};
    float row_sum_error{};
};

SoftmaxMetrics test_softmax() {
    using namespace vortex;

    constexpr std::size_t ROWS =
        4;

    constexpr std::size_t COLUMNS =
        17;

    std::vector<float> input(
        ROWS *
        COLUMNS
    );

    for (
        std::size_t column = 0;
        column < COLUMNS;
        ++column
    ) {
        input[column] =
            10000.0f -
            static_cast<float>(
                column
            ) *
            0.5f;

        input[
            COLUMNS +
            column
        ] =
            -10000.0f +
            static_cast<float>(
                column
            ) *
            0.25f;

        input[
            2 *
            COLUMNS +
            column
        ] =
            42.0f;

        input[
            3 *
            COLUMNS +
            column
        ] =
            (
                column % 2 == 0
            )
                ? 80.0f
                : -80.0f;
    }

    Tensor device_input(
        {ROWS, COLUMNS},
        DataType::Float32,
        Device::CUDA
    );

    Tensor standard_output(
        {ROWS, COLUMNS},
        DataType::Float32,
        Device::CUDA
    );

    Tensor scaled_output(
        {ROWS, COLUMNS},
        DataType::Float32,
        Device::CUDA
    );

    device_input.copy_from_host(
        input.data(),
        input.size() *
            sizeof(float)
    );

    ops::softmax(
        device_input,
        standard_output
    );

    constexpr float SCALE =
        0.125f;

    ops::scaled_softmax(
        device_input,
        SCALE,
        scaled_output
    );

    require(
        numerics::all_finite(
            standard_output
        ),
        "Standard Softmax produced NaN or Inf"
    );

    require(
        numerics::all_finite(
            scaled_output
        ),
        "Scaled Softmax produced NaN or Inf"
    );

    std::vector<float> standard_host(
        input.size()
    );

    std::vector<float> scaled_host(
        input.size()
    );

    standard_output.copy_to_host(
        standard_host.data(),
        standard_host.size() *
            sizeof(float)
    );

    scaled_output.copy_to_host(
        scaled_host.data(),
        scaled_host.size() *
            sizeof(float)
    );

    SoftmaxMetrics metrics;

    for (
        std::size_t row = 0;
        row < ROWS;
        ++row
    ) {
        double maximum =
            -std::numeric_limits<double>::
                infinity();

        double scaled_maximum =
            -std::numeric_limits<double>::
                infinity();

        for (
            std::size_t column = 0;
            column < COLUMNS;
            ++column
        ) {
            const double value =
                input[
                    row *
                    COLUMNS +
                    column
                ];

            maximum =
                std::max(
                    maximum,
                    value
                );

            scaled_maximum =
                std::max(
                    scaled_maximum,
                    value *
                    SCALE
                );
        }

        double denominator =
            0.0;

        double scaled_denominator =
            0.0;

        for (
            std::size_t column = 0;
            column < COLUMNS;
            ++column
        ) {
            const double value =
                input[
                    row *
                    COLUMNS +
                    column
                ];

            denominator +=
                std::exp(
                    value -
                    maximum
                );

            scaled_denominator +=
                std::exp(
                    value *
                    SCALE -
                    scaled_maximum
                );
        }

        float standard_sum =
            0.0f;

        float scaled_sum =
            0.0f;

        for (
            std::size_t column = 0;
            column < COLUMNS;
            ++column
        ) {
            const std::size_t index =
                row *
                COLUMNS +
                column;

            const double value =
                input[index];

            const float standard_reference =
                static_cast<float>(
                    std::exp(
                        value -
                        maximum
                    ) /
                    denominator
                );

            const float scaled_reference =
                static_cast<float>(
                    std::exp(
                        value *
                        SCALE -
                        scaled_maximum
                    ) /
                    scaled_denominator
                );

            metrics.standard_error =
                std::max(
                    metrics.standard_error,
                    std::fabs(
                        standard_host[index] -
                        standard_reference
                    )
                );

            metrics.scaled_error =
                std::max(
                    metrics.scaled_error,
                    std::fabs(
                        scaled_host[index] -
                        scaled_reference
                    )
                );

            standard_sum +=
                standard_host[index];

            scaled_sum +=
                scaled_host[index];
        }

        metrics.row_sum_error =
            std::max(
                metrics.row_sum_error,
                std::fabs(
                    standard_sum -
                    1.0f
                )
            );

        metrics.row_sum_error =
            std::max(
                metrics.row_sum_error,
                std::fabs(
                    scaled_sum -
                    1.0f
                )
            );
    }

    return metrics;
}

// ============================================================
// LayerNorm through GraphExecutor
// ============================================================

float run_layernorm_case(
    const std::vector<std::size_t>& shape,
    unsigned int seed
) {
    using namespace vortex;

    const std::size_t features =
        shape.back();

    std::size_t elements =
        1;

    for (
        const std::size_t dimension :
        shape
    ) {
        elements *=
            dimension;
    }

    const std::size_t rows =
        elements /
        features;

    std::mt19937 generator(
        seed
    );

    std::uniform_real_distribution<float>
        input_noise(
            -0.5f,
            0.5f
        );

    std::uniform_real_distribution<float>
        gamma_distribution(
            0.8f,
            1.2f
        );

    std::uniform_real_distribution<float>
        beta_distribution(
            -0.2f,
            0.2f
        );

    std::vector<float> input(
        elements
    );

    std::vector<float> gamma(
        features
    );

    std::vector<float> beta(
        features
    );

    for (float& value : input) {
        value =
            1000.0f +
            input_noise(
                generator
            );
    }

    for (
        std::size_t feature = 0;
        feature < features;
        ++feature
    ) {
        gamma[feature] =
            gamma_distribution(
                generator
            );

        beta[feature] =
            beta_distribution(
                generator
            );
    }

    std::vector<float> reference(
        elements
    );

    constexpr double EPSILON =
        1e-5;

    for (
        std::size_t row = 0;
        row < rows;
        ++row
    ) {
        double mean =
            0.0;

        for (
            std::size_t feature = 0;
            feature < features;
            ++feature
        ) {
            mean +=
                input[
                    row *
                    features +
                    feature
                ];
        }

        mean /=
            static_cast<double>(
                features
            );

        double variance =
            0.0;

        for (
            std::size_t feature = 0;
            feature < features;
            ++feature
        ) {
            const double difference =
                input[
                    row *
                    features +
                    feature
                ] -
                mean;

            variance +=
                difference *
                difference;
        }

        variance /=
            static_cast<double>(
                features
            );

        const double inverse_std =
            1.0 /
            std::sqrt(
                variance +
                EPSILON
            );

        for (
            std::size_t feature = 0;
            feature < features;
            ++feature
        ) {
            const std::size_t index =
                row *
                features +
                feature;

            reference[index] =
                static_cast<float>(
                    (
                        input[index] -
                        mean
                    ) *
                    inverse_std *
                    gamma[feature] +
                    beta[feature]
                );
        }
    }

    Graph graph;

    const TensorId input_id =
        graph.add_tensor(
            "input",
            shape,
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId gamma_id =
        graph.add_tensor(
            "gamma",
            {features},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId beta_id =
        graph.add_tensor(
            "beta",
            {features},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId output_id =
        graph.add_tensor(
            "output",
            shape,
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "layernorm",
        OpType::LayerNorm,
        {
            input_id,
            gamma_id,
            beta_id
        },
        {output_id}
    );

    graph.mark_output(
        output_id
    );

    GraphExecutor executor(
        graph
    );

    executor.tensor(
        input_id
    ).copy_from_host(
        input.data(),
        input.size() *
            sizeof(float)
    );

    executor.tensor(
        gamma_id
    ).copy_from_host(
        gamma.data(),
        gamma.size() *
            sizeof(float)
    );

    executor.tensor(
        beta_id
    ).copy_from_host(
        beta.data(),
        beta.size() *
            sizeof(float)
    );

    executor.run();

    require(
        numerics::all_finite(
            executor.tensor(
                output_id
            )
        ),
        "LayerNorm produced NaN or Inf"
    );

    std::vector<float> result(
        elements
    );

    executor.tensor(
        output_id
    ).copy_to_host(
        result.data(),
        result.size() *
            sizeof(float)
    );

    return compare(
        result,
        reference
    ).maximum;
}

// ============================================================
// Attention
// ============================================================

struct AttentionMetrics {
    float maximum{};
    double mean{};
    float repeat_error{};
};

AttentionMetrics run_attention_case(
    std::size_t batch,
    std::size_t tokens,
    std::size_t dimension,
    unsigned int seed,
    float magnitude,
    bool test_repeated_execution
) {
    using namespace vortex;

    const std::size_t elements =
        batch *
        tokens *
        dimension;

    std::mt19937 generator(
        seed
    );

    std::normal_distribution<float>
        distribution(
            0.0f,
            magnitude
        );

    std::vector<float> query(
        elements
    );

    std::vector<float> key(
        elements
    );

    std::vector<float> value(
        elements
    );

    for (
        std::size_t index = 0;
        index < elements;
        ++index
    ) {
        query[index] =
            distribution(
                generator
            );

        key[index] =
            distribution(
                generator
            );

        value[index] =
            distribution(
                generator
            );
    }

    std::vector<float> reference(
        elements,
        0.0f
    );

    std::vector<double> scores(
        tokens
    );

    std::vector<double> probabilities(
        tokens
    );

    const double scale =
        1.0 /
        std::sqrt(
            static_cast<double>(
                dimension
            )
        );

    for (
        std::size_t batch_index = 0;
        batch_index < batch;
        ++batch_index
    ) {
        for (
            std::size_t query_token = 0;
            query_token < tokens;
            ++query_token
        ) {
            double maximum =
                -std::numeric_limits<double>::
                    infinity();

            for (
                std::size_t key_token = 0;
                key_token < tokens;
                ++key_token
            ) {
                double dot =
                    0.0;

                for (
                    std::size_t feature = 0;
                    feature < dimension;
                    ++feature
                ) {
                    dot +=
                        static_cast<double>(
                            query[
                                batch_index *
                                    tokens *
                                    dimension +
                                query_token *
                                    dimension +
                                feature
                            ]
                        ) *
                        static_cast<double>(
                            key[
                                batch_index *
                                    tokens *
                                    dimension +
                                key_token *
                                    dimension +
                                feature
                            ]
                        );
                }

                scores[key_token] =
                    dot *
                    scale;

                maximum =
                    std::max(
                        maximum,
                        scores[key_token]
                    );
            }

            double denominator =
                0.0;

            for (
                std::size_t key_token = 0;
                key_token < tokens;
                ++key_token
            ) {
                probabilities[key_token] =
                    std::exp(
                        scores[key_token] -
                        maximum
                    );

                denominator +=
                    probabilities[key_token];
            }

            for (
                double& probability :
                probabilities
            ) {
                probability /=
                    denominator;
            }

            for (
                std::size_t feature = 0;
                feature < dimension;
                ++feature
            ) {
                double context =
                    0.0;

                for (
                    std::size_t key_token = 0;
                    key_token < tokens;
                    ++key_token
                ) {
                    context +=
                        probabilities[key_token] *
                        value[
                            batch_index *
                                tokens *
                                dimension +
                            key_token *
                                dimension +
                            feature
                        ];
                }

                reference[
                    batch_index *
                        tokens *
                        dimension +
                    query_token *
                        dimension +
                    feature
                ] =
                    static_cast<float>(
                        context
                    );
            }
        }
    }

    Tensor device_query(
        {
            batch,
            tokens,
            dimension
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor device_key(
        {
            batch,
            tokens,
            dimension
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor device_value(
        {
            batch,
            tokens,
            dimension
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor output(
        {
            batch,
            tokens,
            dimension
        },
        DataType::Float32,
        Device::CUDA
    );

    device_query.copy_from_host(
        query.data(),
        query.size() *
            sizeof(float)
    );

    device_key.copy_from_host(
        key.data(),
        key.size() *
            sizeof(float)
    );

    device_value.copy_from_host(
        value.data(),
        value.size() *
            sizeof(float)
    );

    ops::scaled_dot_product_attention(
        device_query,
        device_key,
        device_value,
        output
    );

    require(
        numerics::all_finite(
            output
        ),
        "Attention produced NaN or Inf"
    );

    std::vector<float> result(
        elements
    );

    output.copy_to_host(
        result.data(),
        result.size() *
            sizeof(float)
    );

    const ErrorMetrics comparison =
        compare(
            result,
            reference
        );

    AttentionMetrics metrics{
        comparison.maximum,
        comparison.mean,
        0.0f
    };

    if (test_repeated_execution) {
        const std::vector<float> first =
            result;

        for (
            int iteration = 0;
            iteration < 25;
            ++iteration
        ) {
            ops::scaled_dot_product_attention(
                device_query,
                device_key,
                device_value,
                output
            );
        }

        output.copy_to_host(
            result.data(),
            result.size() *
                sizeof(float)
        );

        metrics.repeat_error =
            compare(
                result,
                first
            ).maximum;
    }

    return metrics;
}

// ============================================================
// FP16
// ============================================================

struct FP16Metrics {
    float relu_error{};
    float gelu_error{};
};

FP16Metrics test_fp16() {
    using namespace vortex;

    constexpr std::size_t ELEMENTS =
        4096;

    std::vector<float> input_float(
        ELEMENTS
    );

    std::vector<__half> input_half(
        ELEMENTS
    );

    for (
        std::size_t index = 0;
        index < ELEMENTS;
        ++index
    ) {
        input_float[index] =
            -8.0f +
            16.0f *
            static_cast<float>(
                index
            ) /
            static_cast<float>(
                ELEMENTS -
                1
            );

        input_half[index] =
            __float2half(
                input_float[index]
            );
    }

    Tensor input(
        {ELEMENTS},
        DataType::Float16,
        Device::CUDA
    );

    Tensor relu_output(
        {ELEMENTS},
        DataType::Float16,
        Device::CUDA
    );

    Tensor gelu_output(
        {ELEMENTS},
        DataType::Float16,
        Device::CUDA
    );

    input.copy_from_host(
        input_half.data(),
        input_half.size() *
            sizeof(__half)
    );

    ops::relu_fp16(
        input,
        relu_output
    );

    ops::gelu_fp16(
        input,
        gelu_output
    );

    require(
        numerics::all_finite(
            relu_output
        ),
        "FP16 ReLU produced NaN or Inf"
    );

    require(
        numerics::all_finite(
            gelu_output
        ),
        "FP16 GELU produced NaN or Inf"
    );

    std::vector<__half> relu_half(
        ELEMENTS
    );

    std::vector<__half> gelu_half(
        ELEMENTS
    );

    relu_output.copy_to_host(
        relu_half.data(),
        relu_half.size() *
            sizeof(__half)
    );

    gelu_output.copy_to_host(
        gelu_half.data(),
        gelu_half.size() *
            sizeof(__half)
    );

    FP16Metrics metrics;

    for (
        std::size_t index = 0;
        index < ELEMENTS;
        ++index
    ) {
        const float quantized_input =
            __half2float(
                input_half[index]
            );

        const float relu_reference =
            std::max(
                0.0f,
                quantized_input
            );

        const float gelu_value =
            static_cast<float>(
                gelu_reference(
                    quantized_input
                )
            );

        metrics.relu_error =
            std::max(
                metrics.relu_error,
                std::fabs(
                    __half2float(
                        relu_half[index]
                    ) -
                    relu_reference
                )
            );

        metrics.gelu_error =
            std::max(
                metrics.gelu_error,
                std::fabs(
                    __half2float(
                        gelu_half[index]
                    ) -
                    gelu_value
                )
            );
    }

    return metrics;
}

// ============================================================
// Explicit finite-value detection
// ============================================================

bool test_finite_detection() {
    using namespace vortex;

    std::vector<float> finite_values{
        -10.0f,
        0.0f,
        1.0f,
        5000.0f
    };

    std::vector<float> nan_values =
        finite_values;

    std::vector<float> infinity_values =
        finite_values;

    nan_values[1] =
        std::numeric_limits<float>::
            quiet_NaN();

    infinity_values[2] =
        std::numeric_limits<float>::
            infinity();

    Tensor finite_tensor(
        {finite_values.size()},
        DataType::Float32,
        Device::CUDA
    );

    Tensor nan_tensor(
        {nan_values.size()},
        DataType::Float32,
        Device::CUDA
    );

    Tensor infinity_tensor(
        {infinity_values.size()},
        DataType::Float32,
        Device::CUDA
    );

    finite_tensor.copy_from_host(
        finite_values.data(),
        finite_values.size() *
            sizeof(float)
    );

    nan_tensor.copy_from_host(
        nan_values.data(),
        nan_values.size() *
            sizeof(float)
    );

    infinity_tensor.copy_from_host(
        infinity_values.data(),
        infinity_values.size() *
            sizeof(float)
    );

    return
        numerics::all_finite(
            finite_tensor
        ) &&
        !numerics::all_finite(
            nan_tensor
        ) &&
        !numerics::all_finite(
            infinity_tensor
        );
}

} // namespace

int main() {
    try {
        const SoftmaxMetrics softmax =
            test_softmax();

        float layernorm_error =
            0.0f;

        layernorm_error =
            std::max(
                layernorm_error,
                run_layernorm_case(
                    {5, 31},
                    2501
                )
            );

        layernorm_error =
            std::max(
                layernorm_error,
                run_layernorm_case(
                    {2, 7, 32},
                    2502
                )
            );

        AttentionMetrics attention;

        const std::vector<
            AttentionMetrics
        > attention_cases{
            run_attention_case(
                1,
                1,
                8,
                2511,
                0.25f,
                false
            ),

            run_attention_case(
                2,
                7,
                13,
                2512,
                0.5f,
                true
            ),

            run_attention_case(
                3,
                16,
                32,
                2513,
                2.0f,
                false
            )
        };

        for (
            const AttentionMetrics& current :
            attention_cases
        ) {
            attention.maximum =
                std::max(
                    attention.maximum,
                    current.maximum
                );

            attention.mean =
                std::max(
                    attention.mean,
                    current.mean
                );

            attention.repeat_error =
                std::max(
                    attention.repeat_error,
                    current.repeat_error
                );
        }

        const FP16Metrics fp16 =
            test_fp16();

        const bool finite_detection =
            test_finite_detection();

        std::cout
            << "=================================\n"
            << " VORTEX Phase-25 Numerical Validation\n"
            << "=================================\n\n";

        std::cout
            << "softmax_max_error="
            << softmax.standard_error
            << '\n';

        std::cout
            << "scaled_softmax_max_error="
            << softmax.scaled_error
            << '\n';

        std::cout
            << "softmax_row_sum_error="
            << softmax.row_sum_error
            << '\n';

        std::cout
            << "layernorm_max_error="
            << layernorm_error
            << '\n';

        std::cout
            << "attention_max_error="
            << attention.maximum
            << '\n';

        std::cout
            << "attention_mean_error="
            << attention.mean
            << '\n';

        std::cout
            << "attention_repeat_error="
            << attention.repeat_error
            << '\n';

        std::cout
            << "fp16_relu_max_error="
            << fp16.relu_error
            << '\n';

        std::cout
            << "fp16_gelu_max_error="
            << fp16.gelu_error
            << '\n';

        std::cout
            << "finite_detection="
            << (
                finite_detection
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        const bool passed =
            softmax.standard_error <=
                2e-5f &&
            softmax.scaled_error <=
                2e-5f &&
            softmax.row_sum_error <=
                2e-5f &&
            layernorm_error <=
                3e-3f &&
            attention.maximum <=
                8e-4f &&
            attention.mean <=
                8e-5 &&
            attention.repeat_error <=
                1e-7f &&
            fp16.relu_error <=
                2e-3f &&
            fp16.gelu_error <=
                1e-2f &&
            finite_detection;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Phase-25 numerical validation FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Phase-25 numerical validation PASSED\n";

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
