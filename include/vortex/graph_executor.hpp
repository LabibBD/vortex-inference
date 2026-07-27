#pragma once

#include "vortex/graph.hpp"
#include "vortex/tensor.hpp"

#include <memory>
#include <vector>

namespace vortex {

class GraphExecutor {
public:
    explicit GraphExecutor(
        const Graph& graph
    );

    GraphExecutor(
        const GraphExecutor&
    ) = delete;

    GraphExecutor&
    operator=(
        const GraphExecutor&
    ) = delete;

    Tensor& tensor(
        TensorId id
    );

    const Tensor& tensor(
        TensorId id
    ) const;

    void run();

private:
    void execute_node(
        const Node& node
    );

    const Graph& graph_;

    std::vector<
        std::unique_ptr<Tensor>
    > tensors_;
};

} // namespace vortex
