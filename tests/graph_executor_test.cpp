#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t BATCH = 8;
    constexpr std::size_t INPUT = 16;
    constexpr std::size_t HIDDEN = 32;
    constexpr std::size_t CLASSES = 7;

    Graph graph;

    const TensorId input =
        graph.add_tensor(
            "input",
            {BATCH, INPUT},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId w1 =
        graph.add_tensor(
            "w1",
            {INPUT, HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId b1 =
        graph.add_tensor(
            "b1",
            {HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId hidden =
        graph.add_tensor(
            "hidden",
            {BATCH, HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "linear1",
        OpType::Linear,
        {input, w1, b1},
        {hidden}
    );

    const TensorId activated =
        graph.add_tensor(
            "activated",
            {BATCH, HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "gelu",
        OpType::GELU,
        {hidden},
        {activated}
    );

    const TensorId w2 =
        graph.add_tensor(
            "w2",
            {HIDDEN, CLASSES},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId b2 =
        graph.add_tensor(
            "b2",
            {CLASSES},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId logits =
        graph.add_tensor(
            "logits",
            {BATCH, CLASSES},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "linear2",
        OpType::Linear,
        {activated, w2, b2},
        {logits}
    );

    const TensorId probabilities =
        graph.add_tensor(
            "probabilities",
            {BATCH, CLASSES},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "softmax",
        OpType::Softmax,
        {logits},
        {probabilities}
    );

    graph.mark_output(
        probabilities
    );

    GraphExecutor executor(
        graph
    );

    // ==================================================
    // Deterministic model data
    // ==================================================

    std::vector<float> h_input(
        BATCH * INPUT
    );

    std::vector<float> h_w1(
        INPUT * HIDDEN
    );

    std::vector<float> h_b1(
        HIDDEN
    );

    std::vector<float> h_w2(
        HIDDEN * CLASSES
    );

    std::vector<float> h_b2(
        CLASSES
    );

    for (
        std::size_t i = 0;
        i < h_input.size();
        ++i
    ) {
        h_input[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.03f;
    }

    for (
        std::size_t i = 0;
        i < h_w1.size();
        ++i
    ) {
        h_w1[i] =
            static_cast<float>(
                static_cast<int>(i % 13) - 6
            ) * 0.02f;
    }

    for (
        std::size_t i = 0;
        i < h_b1.size();
        ++i
    ) {
        h_b1[i] =
            static_cast<float>(
                static_cast<int>(i % 5) - 2
            ) * 0.01f;
    }

    for (
        std::size_t i = 0;
        i < h_w2.size();
        ++i
    ) {
        h_w2[i] =
            static_cast<float>(
                static_cast<int>(i % 11) - 5
            ) * 0.025f;
    }

    for (
        std::size_t i = 0;
        i < h_b2.size();
        ++i
    ) {
        h_b2[i] =
            static_cast<float>(
                static_cast<int>(i) - 3
            ) * 0.01f;
    }

    executor.tensor(input).copy_from_host(
        h_input.data(),
        h_input.size() * sizeof(float)
    );

    executor.tensor(w1).copy_from_host(
        h_w1.data(),
        h_w1.size() * sizeof(float)
    );

    executor.tensor(b1).copy_from_host(
        h_b1.data(),
        h_b1.size() * sizeof(float)
    );

    executor.tensor(w2).copy_from_host(
        h_w2.data(),
        h_w2.size() * sizeof(float)
    );

    executor.tensor(b2).copy_from_host(
        h_b2.data(),
        h_b2.size() * sizeof(float)
    );

    // ==================================================
    // Execute graph
    // ==================================================

    executor.run();

    std::vector<float> result(
        BATCH * CLASSES
    );

    executor.tensor(
        probabilities
    ).copy_to_host(
        result.data(),
        result.size() * sizeof(float)
    );

    // ==================================================
    // Validation
    // Each Softmax row must sum to ~1.
    // ==================================================

    float max_sum_error =
        0.0f;

    bool valid_probabilities =
        true;

    for (
        std::size_t row = 0;
        row < BATCH;
        ++row
    ) {
        float sum =
            0.0f;

        for (
            std::size_t col = 0;
            col < CLASSES;
            ++col
        ) {
            const float value =
                result[
                    row * CLASSES +
                    col
                ];

            if (
                !std::isfinite(value) ||
                value < 0.0f ||
                value > 1.0f
            ) {
                valid_probabilities =
                    false;
            }

            sum += value;
        }

        max_sum_error =
            std::max(
                max_sum_error,
                std::fabs(
                    sum - 1.0f
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX Graph Executor\n"
        << "=================================\n\n";

    std::cout
        << "Nodes executed:     "
        << graph.nodes().size()
        << '\n';

    std::cout
        << "Batch size:         "
        << BATCH
        << '\n';

    std::cout
        << "Output classes:     "
        << CLASSES
        << '\n';

    std::cout
        << "Max row-sum error:  "
        << max_sum_error
        << '\n';

    std::cout
        << "Probabilities valid:"
        << " "
        << (
            valid_probabilities
                ? "YES"
                : "NO"
        )
        << '\n';

    const bool passed =
        valid_probabilities &&
        max_sum_error <= 1e-5f;

    if (passed) {
        std::cout
            << "\n[VORTEX] Graph execution PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Graph execution FAILED\n";

    return 1;
}
