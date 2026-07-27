#include "vortex/graph_executor.hpp"

#include "vortex/operators/activations.hpp"
#include "vortex/operators/elementwise.hpp"
#include "vortex/operators/layernorm.hpp"
#include "vortex/operators/linear.hpp"
#include "vortex/operators/matmul.hpp"
#include "vortex/operators/softmax.hpp"

#include <memory>
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
    : graph_(graph),
      lifetimes_(
          analyze_tensor_lifetimes(graph)
      ) {

    graph_.validate();

    tensors_.resize(
        graph_.tensors().size()
    );

    /*
     * Inputs and constants must exist immediately
     * because callers load data into them before run().
     *
     * Intermediate/output tensors are allocated lazily.
     */
    for (
        const TensorDesc& desc :
        graph_.tensors()
    ) {
        if (
            desc.role == TensorRole::Input ||
            desc.role == TensorRole::Constant
        ) {
            tensors_[desc.id] =
                std::make_unique<Tensor>(
                    desc.shape,
                    desc.dtype,
                    desc.device
                );
        }
    }
}

Tensor& GraphExecutor::tensor(
    TensorId id
) {
    return live_tensor(id);
}

const Tensor& GraphExecutor::tensor(
    TensorId id
) const {
    return live_tensor(id);
}

Tensor& GraphExecutor::live_tensor(
    TensorId id
) {
    if (id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid GraphExecutor TensorId"
        );
    }

    if (!tensors_[id]) {
        throw std::runtime_error(
            "Requested tensor is not currently live"
        );
    }

    return *tensors_[id];
}

const Tensor& GraphExecutor::live_tensor(
    TensorId id
) const {
    if (id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid GraphExecutor TensorId"
        );
    }

    if (!tensors_[id]) {
        throw std::runtime_error(
            "Requested tensor is not currently live"
        );
    }

    return *tensors_[id];
}

Tensor& GraphExecutor::ensure_tensor(
    TensorId id
) {
    if (id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid GraphExecutor TensorId"
        );
    }

    if (!tensors_[id]) {
        const TensorDesc& desc =
            graph_.tensor(id);

        tensors_[id] =
            std::make_unique<Tensor>(
                desc.shape,
                desc.dtype,
                desc.device
            );
    }

    return *tensors_[id];
}

void GraphExecutor::run() {
    for (
        const Node& node :
        graph_.nodes()
    ) {
        /*
         * Allocate node outputs only when execution
         * actually reaches them.
         */
        for (
            const TensorId output :
            node.outputs
        ) {
            ensure_tensor(
                output
            );
        }

        execute_node(
            node
        );

        /*
         * Inputs whose final consumer was this node
         * can now safely release their GPU storage.
         */
        release_dead_tensors(
            node
        );
    }
}

void GraphExecutor::release_dead_tensors(
    const Node& node
) {
    for (
        const TensorId input :
        node.inputs
    ) {
        const TensorDesc& desc =
            graph_.tensor(input);

        const TensorLifetime& lifetime =
            lifetimes_[input];

        if (
            desc.role ==
                TensorRole::Intermediate &&
            lifetime.last_consumer ==
                node.id
        ) {
            tensors_[input].reset();
        }
    }

    /*
     * A dead intermediate may be produced but never
     * consumed. Release it immediately.
     */
    for (
        const TensorId output :
        node.outputs
    ) {
        const TensorDesc& desc =
            graph_.tensor(output);

        const TensorLifetime& lifetime =
            lifetimes_[output];

        if (
            desc.role ==
                TensorRole::Intermediate &&
            lifetime.consumer_count == 0
        ) {
            tensors_[output].reset();
        }
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
                live_tensor(node.inputs[0]),
                live_tensor(node.inputs[1]),
                live_tensor(node.inputs[2]),
                live_tensor(node.outputs[0])
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
                live_tensor(node.inputs[0]),
                live_tensor(node.inputs[1]),
                live_tensor(node.outputs[0])
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
                live_tensor(node.inputs[0]),
                live_tensor(node.outputs[0])
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
                live_tensor(node.inputs[0]),
                live_tensor(node.outputs[0])
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
                live_tensor(node.inputs[0]),
                live_tensor(node.inputs[1]),
                live_tensor(node.inputs[2]),
                live_tensor(node.outputs[0])
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
                live_tensor(node.inputs[0]),
                live_tensor(node.outputs[0])
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
                live_tensor(node.inputs[0]),
                live_tensor(node.inputs[1]),
                live_tensor(node.outputs[0])
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
