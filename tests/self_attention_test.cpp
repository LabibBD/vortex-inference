#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t BATCH = 2;
    constexpr std::size_t TOKENS = 9;
    constexpr std::size_t HEAD_DIM = 13;

    std::vector<float> query(
        BATCH * TOKENS * HEAD_DIM
    );

    std::vector<float> key(
        BATCH * TOKENS * HEAD_DIM
    );

    std::vector<float> value(
        BATCH * TOKENS * HEAD_DIM
    );

    for (
        std::size_t i = 0;
        i < query.size();
        ++i
    ) {
        query[i] =
            static_cast<float>(
                static_cast<int>(i % 23) - 11
            ) * 0.025f;

        key[i] =
            static_cast<float>(
                static_cast<int>(i % 19) - 9
            ) * 0.02f;

        value[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.03f;
    }

    // ==================================================
    // CPU reference
    // ==================================================

    std::vector<float> reference(
        BATCH * TOKENS * HEAD_DIM,
        0.0f
    );

    std::vector<float> scores(
        TOKENS
    );

    std::vector<float> probabilities(
        TOKENS
    );

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(
                HEAD_DIM
            )
        );

    for (
        std::size_t batch = 0;
        batch < BATCH;
        ++batch
    ) {
        for (
            std::size_t query_token = 0;
            query_token < TOKENS;
            ++query_token
        ) {
            float row_max =
                -std::numeric_limits<float>::infinity();

            for (
                std::size_t key_token = 0;
                key_token < TOKENS;
                ++key_token
            ) {
                float dot =
                    0.0f;

                for (
                    std::size_t feature = 0;
                    feature < HEAD_DIM;
                    ++feature
                ) {
                    dot +=
                        query[
                            batch *
                                TOKENS *
                                HEAD_DIM +
                            query_token *
                                HEAD_DIM +
                            feature
                        ] *
                        key[
                            batch *
                                TOKENS *
                                HEAD_DIM +
                            key_token *
                                HEAD_DIM +
                            feature
                        ];
                }

                scores[key_token] =
                    dot * scale;

                row_max =
                    std::max(
                        row_max,
                        scores[key_token]
                    );
            }

            float row_sum =
                0.0f;

            for (
                std::size_t key_token = 0;
                key_token < TOKENS;
                ++key_token
            ) {
                probabilities[key_token] =
                    std::exp(
                        scores[key_token] -
                        row_max
                    );

                row_sum +=
                    probabilities[key_token];
            }

            for (
                float& probability :
                probabilities
            ) {
                probability /=
                    row_sum;
            }

            for (
                std::size_t feature = 0;
                feature < HEAD_DIM;
                ++feature
            ) {
                float context =
                    0.0f;

                for (
                    std::size_t key_token = 0;
                    key_token < TOKENS;
                    ++key_token
                ) {
                    context +=
                        probabilities[key_token] *
                        value[
                            batch *
                                TOKENS *
                                HEAD_DIM +
                            key_token *
                                HEAD_DIM +
                            feature
                        ];
                }

                reference[
                    batch *
                        TOKENS *
                        HEAD_DIM +
                    query_token *
                        HEAD_DIM +
                    feature
                ] =
                    context;
            }
        }
    }

    // ==================================================
    // VORTEX graph
    // ==================================================

    Graph graph;

    const TensorId query_id =
        graph.add_tensor(
            "query",
            {BATCH, TOKENS, HEAD_DIM},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId key_id =
        graph.add_tensor(
            "key",
            {BATCH, TOKENS, HEAD_DIM},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId value_id =
        graph.add_tensor(
            "value",
            {BATCH, TOKENS, HEAD_DIM},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId output_id =
        graph.add_tensor(
            "context",
            {BATCH, TOKENS, HEAD_DIM},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "scaled_dot_product_attention",
        OpType::SelfAttention,
        {
            query_id,
            key_id,
            value_id
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
        query_id
    ).copy_from_host(
        query.data(),
        query.size() * sizeof(float)
    );

    executor.tensor(
        key_id
    ).copy_from_host(
        key.data(),
        key.size() * sizeof(float)
    );

    executor.tensor(
        value_id
    ).copy_from_host(
        value.data(),
        value.size() * sizeof(float)
    );

    executor.run();

    std::vector<float> output(
        reference.size()
    );

    executor.tensor(
        output_id
    ).copy_to_host(
        output.data(),
        output.size() * sizeof(float)
    );

    float max_error =
        0.0f;

    double mean_error =
        0.0;

    bool finite =
        true;

    for (
        std::size_t i = 0;
        i < output.size();
        ++i
    ) {
        if (!std::isfinite(output[i])) {
            finite = false;
        }

        const float error =
            std::fabs(
                output[i] -
                reference[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );

        mean_error +=
            error;
    }

    mean_error /=
        static_cast<double>(
            output.size()
        );

    std::cout
        << "=================================\n"
        << " VORTEX Self-Attention Test\n"
        << "=================================\n\n";

    std::cout
        << "Batch:              "
        << BATCH << '\n';

    std::cout
        << "Tokens:             "
        << TOKENS << '\n';

    std::cout
        << "Head dimension:     "
        << HEAD_DIM << '\n';

    std::cout
        << "Maximum abs error:  "
        << max_error << '\n';

    std::cout
        << "Mean abs error:     "
        << mean_error << '\n';

    std::cout
        << "All values finite:  "
        << (finite ? "YES" : "NO")
        << '\n';

    const bool passed =
        finite &&
        max_error <= 2e-4f &&
        mean_error <= 2e-5;

    if (passed) {
        std::cout
            << "\n[VORTEX] Self-attention PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Self-attention FAILED\n";

    return 1;
}
