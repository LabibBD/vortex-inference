#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"
#include "vortex/lifetime.hpp"
#include "vortex/memory_pool.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N =
        65536;

    constexpr std::size_t TENSOR_BYTES =
        N * sizeof(float);

    auto& pool =
        MemoryPool::global();

    pool.clear();

    Graph graph;

    const TensorId input =
        graph.add_tensor(
            "input",
            {N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId h1 =
        graph.add_tensor(
            "h1",
            {N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    const TensorId h2 =
        graph.add_tensor(
            "h2",
            {N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    const TensorId h3 =
        graph.add_tensor(
            "h3",
            {N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    const TensorId h4 =
        graph.add_tensor(
            "h4",
            {N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    const TensorId output =
        graph.add_tensor(
            "output",
            {N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "relu1",
        OpType::ReLU,
        {input},
        {h1}
    );

    graph.add_node(
        "gelu1",
        OpType::GELU,
        {h1},
        {h2}
    );

    graph.add_node(
        "relu2",
        OpType::ReLU,
        {h2},
        {h3}
    );

    graph.add_node(
        "gelu2",
        OpType::GELU,
        {h3},
        {h4}
    );

    graph.add_node(
        "relu3",
        OpType::ReLU,
        {h4},
        {output}
    );

    graph.mark_output(
        output
    );

    graph.validate();

    const auto lifetimes =
        analyze_tensor_lifetimes(
            graph
        );

    std::cout
        << "=================================\n"
        << " VORTEX Tensor Lifetime Test\n"
        << "=================================\n\n";

    std::cout
        << "Logical tensors: "
        << graph.tensors().size()
        << '\n';

    std::cout
        << "Graph nodes:     "
        << graph.nodes().size()
        << "\n\n";

    std::cout
        << "Lifetime analysis:\n";

    for (
        const TensorDesc& desc :
        graph.tensors()
    ) {
        const auto& life =
            lifetimes[desc.id];

        std::cout
            << "  "
            << desc.name
            << ": consumers="
            << life.consumer_count;

        if (
            life.last_consumer !=
            NO_NODE
        ) {
            std::cout
                << ", last="
                << life.last_consumer;
        }

        std::cout
            << '\n';
    }

    {
        GraphExecutor executor(
            graph
        );

        std::vector<float> data(
            N,
            0.5f
        );

        executor.tensor(
            input
        ).copy_from_host(
            data.data(),
            data.size() * sizeof(float)
        );

        executor.run();

        std::vector<float> result(
            N
        );

        executor.tensor(
            output
        ).copy_to_host(
            result.data(),
            result.size() * sizeof(float)
        );

        const auto stats =
            pool.stats();

        const std::size_t naive_peak =
            6 * TENSOR_BYTES;

        std::cout
            << "\nMemory results:\n";

        std::cout
            << "Naive all-live bytes: "
            << naive_peak
            << '\n';

        std::cout
            << "Actual peak bytes:    "
            << stats.peak_active_bytes
            << '\n';

        std::cout
            << "CUDA allocations:     "
            << stats.cuda_allocations
            << '\n';

        std::cout
            << "Pool reuse count:     "
            << stats.reuse_count
            << '\n';

        const bool output_valid =
            result[0] >= 0.0f;

        const bool reduced_peak =
            stats.peak_active_bytes <=
            3 * TENSOR_BYTES;

        const bool reused_memory =
            stats.reuse_count >= 1;

        const bool avoided_six_allocations =
            stats.cuda_allocations < 6;

        const bool passed =
            output_valid &&
            reduced_peak &&
            reused_memory &&
            avoided_six_allocations;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Lifetime test FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Lifetime analysis PASSED\n";
    }

    pool.clear();

    return 0;
}
