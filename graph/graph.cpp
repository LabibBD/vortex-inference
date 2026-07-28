#include "vortex/graph.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vortex {

TensorId Graph::add_tensor(
    std::string name,
    std::vector<std::size_t> shape,
    DataType dtype,
    Device device,
    TensorRole role
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Graph tensor name cannot be empty"
        );
    }

    if (shape.empty()) {
        throw std::invalid_argument(
            "Graph tensor shape cannot be empty"
        );
    }

    for (const std::size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument(
                "Graph tensor dimensions must be non-zero"
            );
        }
    }

    const TensorId id =
        tensors_.size();

    tensors_.push_back(
        TensorDesc{
            id,
            std::move(name),
            std::move(shape),
            dtype,
            device,
            role
        }
    );

    return id;
}

NodeId Graph::add_node(
    std::string name,
    OpType op,
    std::vector<TensorId> inputs,
    std::vector<TensorId> outputs
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Graph node name cannot be empty"
        );
    }

    if (inputs.empty()) {
        throw std::invalid_argument(
            "Graph node requires at least one input"
        );
    }

    if (outputs.empty()) {
        throw std::invalid_argument(
            "Graph node requires at least one output"
        );
    }

    const NodeId id =
        nodes_.size();

    nodes_.push_back(
        Node{
            id,
            std::move(name),
            op,
            std::move(inputs),
            std::move(outputs)
        }
    );

    return id;
}

void Graph::mark_output(
    TensorId tensor_id
) {
    if (tensor_id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid graph output TensorId"
        );
    }

    tensors_[tensor_id].role =
        TensorRole::Output;

    outputs_.push_back(
        tensor_id
    );
}

const TensorDesc& Graph::tensor(
    TensorId id
) const {
    if (id >= tensors_.size()) {
        throw std::out_of_range(
            "Invalid TensorId"
        );
    }

    return tensors_[id];
}

const Node& Graph::node(
    NodeId id
) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range(
            "Invalid NodeId"
        );
    }

    return nodes_[id];
}

const std::vector<TensorDesc>&
Graph::tensors() const noexcept {
    return tensors_;
}

const std::vector<Node>&
Graph::nodes() const noexcept {
    return nodes_;
}

const std::vector<TensorId>&
Graph::outputs() const noexcept {
    return outputs_;
}

void Graph::validate() const {
    std::vector<bool> available(
        tensors_.size(),
        false
    );

    std::vector<bool> produced(
        tensors_.size(),
        false
    );

    for (const auto& tensor : tensors_) {
        if (
            tensor.role == TensorRole::Input ||
            tensor.role == TensorRole::Constant
        ) {
            available[tensor.id] = true;
        }
    }

    for (const auto& node : nodes_) {

        for (const TensorId input : node.inputs) {
            if (input >= tensors_.size()) {
                throw std::runtime_error(
                    "Graph node references invalid input TensorId"
                );
            }

            if (!available[input]) {
                throw std::runtime_error(
                    "Graph dependency is not available before node execution"
                );
            }
        }

        for (const TensorId output : node.outputs) {
            if (output >= tensors_.size()) {
                throw std::runtime_error(
                    "Graph node references invalid output TensorId"
                );
            }

            if (produced[output]) {
                throw std::runtime_error(
                    "Tensor has multiple producer nodes"
                );
            }

            if (
                tensors_[output].role ==
                    TensorRole::Input ||
                tensors_[output].role ==
                    TensorRole::Constant
            ) {
                throw std::runtime_error(
                    "Graph node cannot overwrite input or constant tensor"
                );
            }

            produced[output] = true;
            available[output] = true;
        }
    }

    std::unordered_set<TensorId>
        unique_outputs;

    for (const TensorId output : outputs_) {
        if (output >= tensors_.size()) {
            throw std::runtime_error(
                "Invalid graph output TensorId"
            );
        }

        if (!available[output]) {
            throw std::runtime_error(
                "Graph output is unavailable"
            );
        }

        if (!unique_outputs.insert(output).second) {
            throw std::runtime_error(
                "Duplicate graph output"
            );
        }
    }
}

const char* op_type_name(
    OpType type
) {
    switch (type) {
        case OpType::Linear:
            return "Linear";

        case OpType::LinearGELU:
            return "LinearGELU";

        case OpType::Add:
            return "Add";

        case OpType::ReLU:
            return "ReLU";

        case OpType::GELU:
            return "GELU";

        case OpType::LayerNorm:
            return "LayerNorm";

        case OpType::Softmax:
            return "Softmax";

        case OpType::MatMul:
            return "MatMul";

        case OpType::Transpose:
            return "Transpose";

        case OpType::BatchedMatMul:
            return "BatchedMatMul";
    }

    return "Unknown";
}

} // namespace vortex
