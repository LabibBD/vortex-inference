#include "vortex/model.hpp"

#include "vortex/graph_executor.hpp"

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vortex {

namespace {

std::vector<std::string>
split(
    const std::string& text,
    char delimiter
) {
    std::vector<std::string> result;

    std::stringstream stream(
        text
    );

    std::string item;

    while (
        std::getline(
            stream,
            item,
            delimiter
        )
    ) {
        result.push_back(
            item
        );
    }

    return result;
}

std::vector<std::size_t>
parse_shape(
    const std::string& text
) {
    std::vector<std::size_t> shape;

    if (text.empty()) {
        return shape;
    }

    for (
        const std::string& item :
        split(text, ',')
    ) {
        shape.push_back(
            static_cast<std::size_t>(
                std::stoull(item)
            )
        );
    }

    return shape;
}

std::vector<TensorId>
parse_ids(
    const std::string& text
) {
    std::vector<TensorId> ids;

    if (text.empty()) {
        return ids;
    }

    for (
        const std::string& item :
        split(text, ',')
    ) {
        ids.push_back(
            static_cast<TensorId>(
                std::stoull(item)
            )
        );
    }

    return ids;
}

TensorRole parse_role(
    const std::string& value
) {
    if (value == "Input") {
        return TensorRole::Input;
    }

    if (value == "Constant") {
        return TensorRole::Constant;
    }

    if (value == "Intermediate") {
        return TensorRole::Intermediate;
    }

    if (value == "Output") {
        return TensorRole::Output;
    }

    throw std::runtime_error(
        "Unknown VORTEX tensor role: " +
        value
    );
}

DataType parse_dtype(
    const std::string& value
) {
    if (value == "FP32") {
        return DataType::Float32;
    }

    if (value == "FP16") {
        return DataType::Float16;
    }

    throw std::runtime_error(
        "Unknown VORTEX dtype: " +
        value
    );
}

Device parse_device(
    const std::string& value
) {
    if (value == "CUDA") {
        return Device::CUDA;
    }

    if (value == "CPU") {
        return Device::CPU;
    }

    throw std::runtime_error(
        "Unknown VORTEX device: " +
        value
    );
}

OpType parse_op(
    const std::string& value
) {
    if (value == "Linear") {
        return OpType::Linear;
    }

    if (value == "LinearGELU") {
        return OpType::LinearGELU;
    }

    if (value == "Add") {
        return OpType::Add;
    }

    if (value == "ReLU") {
        return OpType::ReLU;
    }

    if (value == "GELU") {
        return OpType::GELU;
    }

    if (value == "LayerNorm") {
        return OpType::LayerNorm;
    }

    if (value == "Softmax") {
        return OpType::Softmax;
    }

    if (value == "MatMul") {
        return OpType::MatMul;
    }

    if (value == "Transpose") {
        return OpType::Transpose;
    }

    if (value == "BatchedMatMul") {
        return OpType::BatchedMatMul;
    }

    if (value == "SelfAttention") {
        return OpType::SelfAttention;
    }

    throw std::runtime_error(
        "Unsupported VORTEX model operator: " +
        value
    );
}

} // namespace

const Graph&
Model::graph() const noexcept {
    return graph_;
}

const std::vector<TensorId>&
Model::inputs() const noexcept {
    return inputs_;
}

const std::vector<TensorId>&
Model::outputs() const noexcept {
    return outputs_;
}

void Model::load_constants(
    GraphExecutor& executor
) const {
    std::ifstream weights(
        weights_path_,
        std::ios::binary
    );

    if (!weights) {
        throw std::runtime_error(
            "Could not open VORTEX weight file: " +
            weights_path_.string()
        );
    }

    for (
        const ConstantRecord& record :
        constants_
    ) {
        Tensor& tensor =
            executor.tensor(
                record.tensor_id
            );

        if (
            tensor.bytes() !=
            record.bytes
        ) {
            throw std::runtime_error(
                "Constant byte count does not "
                "match tensor metadata"
            );
        }

        std::vector<std::byte> buffer(
            record.bytes
        );

        weights.seekg(
            static_cast<std::streamoff>(
                record.offset
            ),
            std::ios::beg
        );

        if (!weights) {
            throw std::runtime_error(
                "Failed to seek VORTEX weight file"
            );
        }

        weights.read(
            reinterpret_cast<char*>(
                buffer.data()
            ),
            static_cast<std::streamsize>(
                buffer.size()
            )
        );

        if (
            weights.gcount() !=
            static_cast<std::streamsize>(
                buffer.size()
            )
        ) {
            throw std::runtime_error(
                "Unexpected end of VORTEX weight file"
            );
        }

        tensor.copy_from_host(
            buffer.data(),
            buffer.size()
        );
    }
}

Model ModelLoader::load(
    const std::filesystem::path&
        model_directory
) {
    const auto manifest_path =
        model_directory /
        "model.vtx";

    std::ifstream manifest(
        manifest_path
    );

    if (!manifest) {
        throw std::runtime_error(
            "Could not open VORTEX manifest: " +
            manifest_path.string()
        );
    }

    Model model;

    std::string weights_filename;

    std::vector<TensorId>
        declared_outputs;

    std::string line;

    bool version_seen =
        false;

    while (
        std::getline(
            manifest,
            line
        )
    ) {
        if (line.empty()) {
            continue;
        }

        const auto fields =
            split(
                line,
                '\t'
            );

        if (fields.empty()) {
            continue;
        }

        // ==============================================
        // Model version
        // ==============================================

        if (
            fields[0] ==
            "VORTEX_MODEL"
        ) {
            if (
                fields.size() != 2 ||
                fields[1] != "1"
            ) {
                throw std::runtime_error(
                    "Unsupported VORTEX model version"
                );
            }

            version_seen =
                true;

            continue;
        }

        // ==============================================
        // Weight file
        // ==============================================

        if (
            fields[0] ==
            "WEIGHTS"
        ) {
            if (fields.size() != 2) {
                throw std::runtime_error(
                    "Malformed WEIGHTS record"
                );
            }

            weights_filename =
                fields[1];

            continue;
        }

        // Counts are informational.
        if (
            fields[0] == "TENSORS" ||
            fields[0] == "NODES"
        ) {
            continue;
        }

        // ==============================================
        // Tensor
        // ==============================================

        if (
            fields[0] ==
            "TENSOR"
        ) {
            if (fields.size() != 10) {
                throw std::runtime_error(
                    "Malformed TENSOR record"
                );
            }

            const TensorId expected_id =
                static_cast<TensorId>(
                    std::stoull(
                        fields[1]
                    )
                );

            const std::string& name =
                fields[2];

            const TensorRole role =
                parse_role(
                    fields[3]
                );

            const DataType dtype =
                parse_dtype(
                    fields[4]
                );

            const Device device =
                parse_device(
                    fields[5]
                );

            const std::size_t rank =
                static_cast<std::size_t>(
                    std::stoull(
                        fields[6]
                    )
                );

            const auto shape =
                parse_shape(
                    fields[7]
                );

            if (
                shape.size() != rank
            ) {
                throw std::runtime_error(
                    "Tensor rank/shape mismatch: " +
                    name
                );
            }

            const TensorId actual_id =
                model.graph_.add_tensor(
                    name,
                    shape,
                    dtype,
                    device,
                    role
                );

            if (
                actual_id !=
                expected_id
            ) {
                throw std::runtime_error(
                    "VORTEX TensorIds are not sequential"
                );
            }

            if (
                role ==
                TensorRole::Input
            ) {
                model.inputs_.push_back(
                    actual_id
                );
            }

            if (
                role ==
                TensorRole::Constant
            ) {
                const long long offset =
                    std::stoll(
                        fields[8]
                    );

                const std::size_t bytes =
                    static_cast<std::size_t>(
                        std::stoull(
                            fields[9]
                        )
                    );

                if (
                    offset < 0 ||
                    bytes == 0
                ) {
                    throw std::runtime_error(
                        "Invalid constant storage record"
                    );
                }

                model.constants_.push_back(
                    ConstantRecord{
                        actual_id,
                        static_cast<std::size_t>(
                            offset
                        ),
                        bytes
                    }
                );
            }

            continue;
        }

        // ==============================================
        // Node
        // ==============================================

        if (
            fields[0] ==
            "NODE"
        ) {
            if (fields.size() != 6) {
                throw std::runtime_error(
                    "Malformed NODE record"
                );
            }

            const NodeId expected_id =
                static_cast<NodeId>(
                    std::stoull(
                        fields[1]
                    )
                );

            const auto inputs =
                parse_ids(
                    fields[4]
                );

            const auto outputs =
                parse_ids(
                    fields[5]
                );

            const NodeId actual_id =
                model.graph_.add_node(
                    fields[2],
                    parse_op(
                        fields[3]
                    ),
                    inputs,
                    outputs
                );

            if (
                actual_id !=
                expected_id
            ) {
                throw std::runtime_error(
                    "VORTEX NodeIds are not sequential"
                );
            }

            continue;
        }

        // ==============================================
        // Graph outputs
        // ==============================================

        if (
            fields[0] ==
            "OUTPUTS"
        ) {
            if (fields.size() != 2) {
                throw std::runtime_error(
                    "Malformed OUTPUTS record"
                );
            }

            declared_outputs =
                parse_ids(
                    fields[1]
                );

            continue;
        }

        throw std::runtime_error(
            "Unknown VORTEX manifest record: " +
            fields[0]
        );
    }

    if (!version_seen) {
        throw std::runtime_error(
            "Missing VORTEX_MODEL version"
        );
    }

    if (weights_filename.empty()) {
        throw std::runtime_error(
            "Missing VORTEX weight filename"
        );
    }

    model.weights_path_ =
        model_directory /
        weights_filename;

    if (
        !std::filesystem::exists(
            model.weights_path_
        )
    ) {
        throw std::runtime_error(
            "VORTEX weights file does not exist"
        );
    }

    for (
        const TensorId output :
        declared_outputs
    ) {
        model.graph_.mark_output(
            output
        );

        model.outputs_.push_back(
            output
        );
    }

    model.graph_.validate();

    if (model.inputs_.empty()) {
        throw std::runtime_error(
            "VORTEX model has no inputs"
        );
    }

    if (model.outputs_.empty()) {
        throw std::runtime_error(
            "VORTEX model has no outputs"
        );
    }

    return model;
}

} // namespace vortex
