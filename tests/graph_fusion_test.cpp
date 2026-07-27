#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"
#include "vortex/graph_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t BATCH = 16;
    constexpr std::size_t INPUT = 64;
    constexpr std::size_t HIDDEN = 128;
    constexpr std::size_t OUTPUT = 32;

    Graph original;

    const TensorId input =
        original.add_tensor(
            "input",
            {BATCH, INPUT},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId w1 =
        original.add_tensor(
            "w1",
            {INPUT, HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId b1 =
        original.add_tensor(
            "b1",
            {HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId linear_output =
        original.add_tensor(
            "linear_output",
            {BATCH, HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    original.add_node(
        "linear1",
        OpType::Linear,
        {input, w1, b1},
        {linear_output}
    );

    const TensorId activated =
        original.add_tensor(
            "activated",
            {BATCH, HIDDEN},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    original.add_node(
        "gelu1",
        OpType::GELU,
        {linear_output},
        {activated}
    );

    const TensorId w2 =
        original.add_tensor(
            "w2",
            {HIDDEN, OUTPUT},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId b2 =
        original.add_tensor(
            "b2",
            {OUTPUT},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId output =
        original.add_tensor(
            "output",
            {BATCH, OUTPUT},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    original.add_node(
        "linear2",
        OpType::Linear,
        {activated, w2, b2},
        {output}
    );

    original.mark_output(
        output
    );

    original.validate();

    OptimizedGraph optimized =
        optimize_graph(
            original
        );

    std::cout
        << "=================================\n"
        << " VORTEX Graph Fusion Optimizer\n"
        << "=================================\n\n";

    std::cout
        << "Original nodes: "
        << optimized.stats.original_nodes
        << '\n';

    std::cout
        << "Optimized nodes:"
        << " "
        << optimized.stats.optimized_nodes
        << '\n';

    std::cout
        << "Linear+GELU fusions: "
        << optimized.stats.fused_linear_gelu
        << "\n\n";

    std::cout
        << "Optimized graph:\n";

    for (
        const Node& node :
        optimized.graph.nodes()
    ) {
        std::cout
            << "  "
            << node.id
            << ": "
            << node.name
            << " ["
            << op_type_name(node.op)
            << "]\n";
    }

    // ==================================================
    // Prepare identical model/input data for both graphs.
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
        HIDDEN * OUTPUT
    );

    std::vector<float> h_b2(
        OUTPUT
    );

    for (std::size_t i = 0; i < h_input.size(); ++i) {
        h_input[i] =
            static_cast<float>(
                static_cast<int>(i % 23) - 11
            ) * 0.02f;
    }

    for (std::size_t i = 0; i < h_w1.size(); ++i) {
        h_w1[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.01f;
    }

    for (std::size_t i = 0; i < h_b1.size(); ++i) {
        h_b1[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            ) * 0.01f;
    }

    for (std::size_t i = 0; i < h_w2.size(); ++i) {
        h_w2[i] =
            static_cast<float>(
                static_cast<int>(i % 13) - 6
            ) * 0.012f;
    }

    for (std::size_t i = 0; i < h_b2.size(); ++i) {
        h_b2[i] =
            static_cast<float>(
                static_cast<int>(i % 5) - 2
            ) * 0.01f;
    }

    GraphExecutor baseline_executor(
        original
    );

    GraphExecutor optimized_executor(
        optimized.graph
    );

    auto load =
        [&](GraphExecutor& executor) {
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
        };

    load(baseline_executor);
    load(optimized_executor);

    baseline_executor.run();
    optimized_executor.run();

    std::vector<float> baseline(
        BATCH * OUTPUT
    );

    std::vector<float> fused(
        BATCH * OUTPUT
    );

    baseline_executor.tensor(
        output
    ).copy_to_host(
        baseline.data(),
        baseline.size() * sizeof(float)
    );

    optimized_executor.tensor(
        output
    ).copy_to_host(
        fused.data(),
        fused.size() * sizeof(float)
    );

    float max_error = 0.0f;
    double mean_error = 0.0;

    for (
        std::size_t i = 0;
        i < baseline.size();
        ++i
    ) {
        const float error =
            std::fabs(
                baseline[i] -
                fused[i]
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
            baseline.size()
        );

    std::cout
        << "\nMaximum output error: "
        << max_error
        << '\n';

    std::cout
        << "Mean output error:    "
        << mean_error
        << '\n';

    const bool passed =
        optimized.stats.original_nodes == 3 &&
        optimized.stats.optimized_nodes == 2 &&
        optimized.stats.fused_linear_gelu == 1 &&
        optimized.graph.nodes()[0].op ==
            OpType::LinearGELU &&
        max_error <= 1e-5f;

    if (passed) {
        std::cout
            << "\n[VORTEX] Automatic graph fusion PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Automatic graph fusion FAILED\n";

    return 1;
}
