#include "vortex/lifetime.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vortex {

std::vector<TensorLifetime>
analyze_tensor_lifetimes(
    const Graph& graph
) {
    std::vector<TensorLifetime> lifetimes(
        graph.tensors().size()
    );

    // ---------------------------------------------
    // Find producer of each tensor.
    // ---------------------------------------------

    for (const Node& node : graph.nodes()) {
        for (const TensorId output : node.outputs) {
            if (output >= lifetimes.size()) {
                throw std::runtime_error(
                    "Invalid graph output TensorId during lifetime analysis"
                );
            }

            if (
                lifetimes[output].producer !=
                NO_NODE
            ) {
                throw std::runtime_error(
                    "Tensor has multiple producers"
                );
            }

            lifetimes[output].producer =
                node.id;
        }
    }

    // ---------------------------------------------
    // Find consumer range.
    // ---------------------------------------------

    for (const Node& node : graph.nodes()) {
        for (const TensorId input : node.inputs) {
            if (input >= lifetimes.size()) {
                throw std::runtime_error(
                    "Invalid graph input TensorId during lifetime analysis"
                );
            }

            TensorLifetime& lifetime =
                lifetimes[input];

            if (
                lifetime.first_consumer ==
                NO_NODE
            ) {
                lifetime.first_consumer =
                    node.id;
            }

            lifetime.last_consumer =
                node.id;

            ++lifetime.consumer_count;
        }
    }

    return lifetimes;
}

} // namespace vortex
