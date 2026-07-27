#pragma once

#include "vortex/graph.hpp"

#include <cstddef>

namespace vortex {

struct GraphOptimizationStats {
    std::size_t original_nodes{0};
    std::size_t optimized_nodes{0};
    std::size_t fused_linear_gelu{0};
};

struct OptimizedGraph {
    Graph graph;
    GraphOptimizationStats stats;
};

OptimizedGraph optimize_graph(
    const Graph& graph
);

} // namespace vortex
