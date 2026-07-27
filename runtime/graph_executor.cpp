#include "vortex/graph_executor.hpp"

#include "vortex/operators/activations.hpp"
#include "vortex/operators/elementwise.hpp"
#include "vortex/operators/layernorm.hpp"
#include "vortex/operators/linear.hpp"
#include "vortex/operators/matmul.hpp"
#include "vortex/operators/softmax.hpp"

#include <stdexcept>

namespace vortex {

namespace {

void require_io(
    const Node& node,
    std::size_t inputs,
    std::size_t outputs
) {
    if (
        node.inputs.size() != inputs ||
        node.outputs.size() != outputs
    ) {
        throw std::runtime_error(
            "Invalid input/output count for graph node: " +
            node.name
        );
    }
}

} // namespace

GraphExecutor::GraphExecutor(
    const Graph& graph
)
    : graph_(graph) {

    graph_.validate();

    tensors_.reserve(
        graph_.tensors().size()
    );

    for (
        const TensorDesc& desc :
        graph_.tensors()
    ) {
        tensors_.push_back(
            std::make_unique<Tensor>(
                desc.shape,
                desc.dtype,
                desc.device
            )
        );
    }
}

Tensor& GraphExecutor::tensor(
    TensorId id
) {
    if (id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid GraphExecutor TensorId"
        );
    }

    return *tensors_[id];
}

const Tensor& GraphExecutor::tensor(
    TensorId id
) const {
    if (id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid GraphExecutor TensorId"
        );
    }

    return *tensors_[id];
}

void GraphExecutor::run() {
    for (
        const Node& node :
        graph_.nodes()
    ) {
        execute_node(
            node
        );
    }
}

void GraphExecutor::execute_node(
    const Node& node
) {
    switch (node.op) {

        case OpType::Linear: {
            require_io(
                node,
                3,
                1
            );

            ops::linear(
                tensor(node.inputs[0]),
                tensor(node.inputs[1]),
                tensor(node.inputs[2]),
                tensor(node.outputs[0])
            );

            break;
        }

        case OpType::Add: {
            require_io(
                node,
                2,
                1
            );

            ops::add(
                tensor(node.inputs[0]),
                tensor(node.inputs[1]),
                tensor(node.outputs[0])
            );

            break;
        }

        case OpType::ReLU: {
            require_io(
                node,
                1,
                1
            );

            ops::relu(
                tensor(node.inputs[0]),
                tensor(node.outputs[0])
            );

            break;
        }

        case OpType::GELU: {
            require_io(
                node,
                1,
                1
            );

            ops::gelu(
                tensor(node.inputs[0]),
                tensor(node.outputs[0])
            );

            break;
        }

        case OpType::LayerNorm: {
            require_io(
                node,
                3,
                1
            );

            ops::layernorm(
                tensor(node.inputs[0]),
                tensor(node.inputs[1]),
                tensor(node.inputs[2]),
                tensor(node.outputs[0])
            );

            break;
        }

        case OpType::Softmax: {
            require_io(
                node,
                1,
                1
            );

            ops::softmax(
                tensor(node.inputs[0]),
                tensor(node.outputs[0])
            );

            break;
        }

        case OpType::MatMul: {
            require_io(
                node,
                2,
                1
            );

            ops::matmul_tiled(
                tensor(node.inputs[0]),
                tensor(node.inputs[1]),
                tensor(node.outputs[0])
            );

            break;
        }

        default:
            throw std::runtime_error(
                "Unsupported graph operator: " +
                node.name
            );
    }
}

} // namespace vortex
