#pragma once

#include "vortex/graph.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace vortex {

constexpr std::size_t NO_NODE =
    std::numeric_limits<std::size_t>::max();

struct TensorLifetime {
    std::size_t producer{NO_NODE};

    std::size_t first_consumer{NO_NODE};
    std::size_t last_consumer{NO_NODE};

    std::size_t consumer_count{0};
};

std::vector<TensorLifetime>
analyze_tensor_lifetimes(
    const Graph& graph
);

} // namespace vortex
