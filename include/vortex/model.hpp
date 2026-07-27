#pragma once

#include "vortex/graph.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace vortex {

class GraphExecutor;

struct ConstantRecord {
    TensorId tensor_id{};
    std::size_t offset{};
    std::size_t bytes{};
};

class Model {
public:
    const Graph& graph() const noexcept;

    const std::vector<TensorId>&
    inputs() const noexcept;

    const std::vector<TensorId>&
    outputs() const noexcept;

    void load_constants(
        GraphExecutor& executor
    ) const;

private:
    friend class ModelLoader;

    Graph graph_;

    std::filesystem::path
        weights_path_;

    std::vector<ConstantRecord>
        constants_;

    std::vector<TensorId>
        inputs_;

    std::vector<TensorId>
        outputs_;
};

class ModelLoader {
public:
    static Model load(
        const std::filesystem::path&
            model_directory
    );
};

} // namespace vortex
