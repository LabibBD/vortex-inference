#pragma once

#include "vortex/tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace vortex {

using TensorId = std::size_t;
using NodeId = std::size_t;

enum class TensorRole {
    Input,
    Constant,
    Intermediate,
    Output
};

enum class OpType {
    Linear,
    LinearGELU,
    Add,
    ReLU,
    GELU,
    LayerNorm,
    Softmax,
    MatMul,
    Transpose,
    BatchedMatMul
};

struct TensorDesc {
    TensorId id{};
    std::string name;
    std::vector<std::size_t> shape;
    DataType dtype{DataType::Float32};
    Device device{Device::CUDA};
    TensorRole role{TensorRole::Intermediate};
};

struct Node {
    NodeId id{};
    std::string name;
    OpType op{OpType::Add};

    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
};

class Graph {
public:
    TensorId add_tensor(
        std::string name,
        std::vector<std::size_t> shape,
        DataType dtype,
        Device device,
        TensorRole role
    );

    NodeId add_node(
        std::string name,
        OpType op,
        std::vector<TensorId> inputs,
        std::vector<TensorId> outputs
    );

    void mark_output(
        TensorId tensor
    );

    const TensorDesc& tensor(
        TensorId id
    ) const;

    const Node& node(
        NodeId id
    ) const;

    const std::vector<TensorDesc>& tensors() const noexcept;

    const std::vector<Node>& nodes() const noexcept;

    const std::vector<TensorId>& outputs() const noexcept;

    void validate() const;

private:
    std::vector<TensorDesc> tensors_;
    std::vector<Node> nodes_;
    std::vector<TensorId> outputs_;
};

const char* op_type_name(
    OpType type
);

} // namespace vortex
