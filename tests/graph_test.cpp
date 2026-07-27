#include "vortex/graph.hpp"

#include <iostream>

int main() {
    using namespace vortex;

    Graph graph;

    const TensorId input =
        graph.add_tensor(
            "input",
            {32, 784},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId w1 =
        graph.add_tensor(
            "linear1.weight",
            {784, 512},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId b1 =
        graph.add_tensor(
            "linear1.bias",
            {512},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId hidden =
        graph.add_tensor(
            "hidden",
            {32, 512},
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
            {32, 512},
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
            "linear2.weight",
            {512, 10},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId b2 =
        graph.add_tensor(
            "linear2.bias",
            {10},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Constant
        );

    const TensorId logits =
        graph.add_tensor(
            "logits",
            {32, 10},
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
            {32, 10},
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

    try {
        graph.validate();
    } catch (const std::exception& error) {
        std::cerr
            << "[VORTEX] Graph validation FAILED: "
            << error.what()
            << '\n';

        return 1;
    }

    std::cout
        << "=================================\n"
        << " VORTEX Computational Graph\n"
        << "=================================\n\n";

    std::cout
        << "Tensors: "
        << graph.tensors().size()
        << '\n';

    std::cout
        << "Nodes:   "
        << graph.nodes().size()
        << '\n';

    std::cout
        << "Outputs: "
        << graph.outputs().size()
        << "\n\n";

    std::cout
        << "Execution order:\n";

    for (const auto& node : graph.nodes()) {
        std::cout
            << "  "
            << node.id
            << ": "
            << node.name
            << " ["
            << op_type_name(node.op)
            << "]\n";
    }

    const bool passed =
        graph.nodes().size() == 4 &&
        graph.outputs().size() == 1 &&
        graph.tensor(
            probabilities
        ).role == TensorRole::Output;

    if (!passed) {
        std::cerr
            << "\n[VORTEX] Graph test FAILED\n";

        return 1;
    }

    std::cout
        << "\n[VORTEX] Graph validation PASSED\n";

    return 0;
}
