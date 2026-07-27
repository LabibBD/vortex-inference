#include "vortex/graph_optimizer.hpp"

#include <cstddef>
#include <vector>

namespace vortex {

OptimizedGraph optimize_graph(
    const Graph& original
) {
    original.validate();

    OptimizedGraph result;

    result.stats.original_nodes =
        original.nodes().size();

    // ==================================================
    // Preserve TensorIds by recreating tensors in exactly
    // the same order.
    // ==================================================

    for (
        const TensorDesc& tensor :
        original.tensors()
    ) {
        result.graph.add_tensor(
            tensor.name,
            tensor.shape,
            tensor.dtype,
            tensor.device,
            tensor.role
        );
    }

    // Number of graph consumers of each tensor.
    std::vector<std::size_t> consumers(
        original.tensors().size(),
        0
    );

    for (
        const Node& node :
        original.nodes()
    ) {
        for (
            const TensorId input :
            node.inputs
        ) {
            ++consumers[input];
        }
    }

    const auto& nodes =
        original.nodes();

    std::size_t i = 0;

    while (i < nodes.size()) {
        const Node& current =
            nodes[i];

        bool fused =
            false;

        // ==================================================
        // Pattern:
        //
        // Linear -> intermediate -> GELU
        //
        // Fusion is legal only when that intermediate has
        // exactly one consumer and is not a graph output.
        // ==================================================

        if (
            current.op == OpType::Linear &&
            current.inputs.size() == 3 &&
            current.outputs.size() == 1 &&
            i + 1 < nodes.size()
        ) {
            const Node& next =
                nodes[i + 1];

            const TensorId intermediate =
                current.outputs[0];

            if (
                next.op == OpType::GELU &&
                next.inputs.size() == 1 &&
                next.outputs.size() == 1 &&
                next.inputs[0] == intermediate &&
                consumers[intermediate] == 1 &&
                original.tensor(intermediate).role ==
                    TensorRole::Intermediate
            ) {
                result.graph.add_node(
                    current.name +
                        "+" +
                        next.name,

                    OpType::LinearGELU,

                    current.inputs,

                    next.outputs
                );

                ++result.stats.fused_linear_gelu;

                i += 2;
                fused = true;
            }
        }

        if (!fused) {
            result.graph.add_node(
                current.name,
                current.op,
                current.inputs,
                current.outputs
            );

            ++i;
        }
    }

    // Restore externally visible graph outputs.
    for (
        const TensorId output :
        original.outputs()
    ) {
        result.graph.mark_output(
            output
        );
    }

    result.graph.validate();

    result.stats.optimized_nodes =
        result.graph.nodes().size();

    return result;
}

} // namespace vortex
