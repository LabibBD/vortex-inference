#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"
#include "vortex/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t BATCH =
    4;

constexpr std::size_t TOKENS =
    16;

constexpr std::size_t MODEL_DIM =
    32;

constexpr std::size_t FF_DIM =
    64;

constexpr int REPEATED_EXECUTIONS =
    20;


struct CaseSpec {
    std::string name;
    unsigned int seed{};
    std::string distribution;
};


struct TransformerIds {
    vortex::TensorId input{};

    vortex::TensorId q_weight{};
    vortex::TensorId q_bias{};

    vortex::TensorId k_weight{};
    vortex::TensorId k_bias{};

    vortex::TensorId v_weight{};
    vortex::TensorId v_bias{};

    vortex::TensorId out_weight{};
    vortex::TensorId out_bias{};

    vortex::TensorId norm1_gamma{};
    vortex::TensorId norm1_beta{};

    vortex::TensorId ff1_weight{};
    vortex::TensorId ff1_bias{};

    vortex::TensorId ff2_weight{};
    vortex::TensorId ff2_bias{};

    vortex::TensorId norm2_gamma{};
    vortex::TensorId norm2_beta{};

    vortex::TensorId output{};
};


std::vector<std::string>
split(
    const std::string& text,
    char delimiter
) {
    std::vector<std::string> fields;

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
        fields.push_back(
            item
        );
    }

    return fields;
}


std::vector<CaseSpec>
read_manifest(
    const std::filesystem::path& path
) {
    std::ifstream file(
        path
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open manifest: " +
            path.string()
        );
    }

    std::vector<CaseSpec> cases;

    std::string line;

    bool header =
        true;

    while (
        std::getline(
            file,
            line
        )
    ) {
        if (line.empty()) {
            continue;
        }

        if (header) {
            header = false;
            continue;
        }

        const auto fields =
            split(
                line,
                '\t'
            );

        if (fields.size() < 3) {
            throw std::runtime_error(
                "Malformed Phase-25 manifest"
            );
        }

        cases.push_back(
            CaseSpec{
                fields[0],
                static_cast<unsigned int>(
                    std::stoul(
                        fields[1]
                    )
                ),
                fields[2]
            }
        );
    }

    if (cases.empty()) {
        throw std::runtime_error(
            "Phase-25 manifest contains no cases"
        );
    }

    return cases;
}


std::vector<float>
read_fp32(
    const std::filesystem::path& path
) {
    std::ifstream file(
        path,
        std::ios::binary |
        std::ios::ate
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open " +
            path.string()
        );
    }

    const auto bytes =
        file.tellg();

    if (
        bytes <= 0 ||
        bytes %
            static_cast<std::streamoff>(
                sizeof(float)
            ) != 0
    ) {
        throw std::runtime_error(
            "Invalid FP32 file: " +
            path.string()
        );
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<float> values(
        static_cast<std::size_t>(
            bytes
        ) /
        sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(
            values.data()
        ),
        bytes
    );

    if (!file) {
        throw std::runtime_error(
            "Failed reading " +
            path.string()
        );
    }

    return values;
}


void load_tensor(
    vortex::GraphExecutor& executor,
    vortex::TensorId tensor_id,
    const std::filesystem::path& path
) {
    const auto values =
        read_fp32(
            path
        );

    vortex::Tensor& tensor =
        executor.tensor(
            tensor_id
        );

    if (
        tensor.bytes() !=
        values.size() *
        sizeof(float)
    ) {
        throw std::runtime_error(
            "Tensor/file size mismatch: " +
            path.string()
        );
    }

    tensor.copy_from_host(
        values.data(),
        values.size() *
            sizeof(float)
    );
}


TransformerIds build_graph(
    vortex::Graph& graph
) {
    using namespace vortex;

    TransformerIds ids;

    const auto model_shape =
        std::vector<std::size_t>{
            BATCH,
            TOKENS,
            MODEL_DIM
        };

    const auto add_constant =
        [&](
            const std::string& name,
            const std::vector<std::size_t>& shape
        ) {
            return graph.add_tensor(
                name,
                shape,
                DataType::Float32,
                Device::CUDA,
                TensorRole::Constant
            );
        };

    const auto add_intermediate =
        [&](
            const std::string& name,
            const std::vector<std::size_t>& shape
        ) {
            return graph.add_tensor(
                name,
                shape,
                DataType::Float32,
                Device::CUDA,
                TensorRole::Intermediate
            );
        };

    ids.input =
        graph.add_tensor(
            "input",
            model_shape,
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    ids.q_weight =
        add_constant(
            "q.weight",
            {
                MODEL_DIM,
                MODEL_DIM
            }
        );

    ids.q_bias =
        add_constant(
            "q.bias",
            {MODEL_DIM}
        );

    ids.k_weight =
        add_constant(
            "k.weight",
            {
                MODEL_DIM,
                MODEL_DIM
            }
        );

    ids.k_bias =
        add_constant(
            "k.bias",
            {MODEL_DIM}
        );

    ids.v_weight =
        add_constant(
            "v.weight",
            {
                MODEL_DIM,
                MODEL_DIM
            }
        );

    ids.v_bias =
        add_constant(
            "v.bias",
            {MODEL_DIM}
        );

    ids.out_weight =
        add_constant(
            "out.weight",
            {
                MODEL_DIM,
                MODEL_DIM
            }
        );

    ids.out_bias =
        add_constant(
            "out.bias",
            {MODEL_DIM}
        );

    ids.norm1_gamma =
        add_constant(
            "norm1.gamma",
            {MODEL_DIM}
        );

    ids.norm1_beta =
        add_constant(
            "norm1.beta",
            {MODEL_DIM}
        );

    ids.ff1_weight =
        add_constant(
            "ff1.weight",
            {
                MODEL_DIM,
                FF_DIM
            }
        );

    ids.ff1_bias =
        add_constant(
            "ff1.bias",
            {FF_DIM}
        );

    ids.ff2_weight =
        add_constant(
            "ff2.weight",
            {
                FF_DIM,
                MODEL_DIM
            }
        );

    ids.ff2_bias =
        add_constant(
            "ff2.bias",
            {MODEL_DIM}
        );

    ids.norm2_gamma =
        add_constant(
            "norm2.gamma",
            {MODEL_DIM}
        );

    ids.norm2_beta =
        add_constant(
            "norm2.beta",
            {MODEL_DIM}
        );

    const TensorId query =
        add_intermediate(
            "query",
            model_shape
        );

    const TensorId key =
        add_intermediate(
            "key",
            model_shape
        );

    const TensorId value =
        add_intermediate(
            "value",
            model_shape
        );

    graph.add_node(
        "query_projection",
        OpType::Linear3D,
        {
            ids.input,
            ids.q_weight,
            ids.q_bias
        },
        {query}
    );

    graph.add_node(
        "key_projection",
        OpType::Linear3D,
        {
            ids.input,
            ids.k_weight,
            ids.k_bias
        },
        {key}
    );

    graph.add_node(
        "value_projection",
        OpType::Linear3D,
        {
            ids.input,
            ids.v_weight,
            ids.v_bias
        },
        {value}
    );

    const TensorId context =
        add_intermediate(
            "context",
            model_shape
        );

    graph.add_node(
        "self_attention",
        OpType::SelfAttention,
        {
            query,
            key,
            value
        },
        {context}
    );

    const TensorId attention_output =
        add_intermediate(
            "attention_output",
            model_shape
        );

    graph.add_node(
        "attention_output_projection",
        OpType::Linear3D,
        {
            context,
            ids.out_weight,
            ids.out_bias
        },
        {attention_output}
    );

    const TensorId residual1 =
        add_intermediate(
            "residual1",
            model_shape
        );

    graph.add_node(
        "attention_residual",
        OpType::Add,
        {
            ids.input,
            attention_output
        },
        {residual1}
    );

    const TensorId normalized1 =
        add_intermediate(
            "normalized1",
            model_shape
        );

    graph.add_node(
        "layernorm1",
        OpType::LayerNorm,
        {
            residual1,
            ids.norm1_gamma,
            ids.norm1_beta
        },
        {normalized1}
    );

    const TensorId hidden =
        add_intermediate(
            "ff_hidden",
            {
                BATCH,
                TOKENS,
                FF_DIM
            }
        );

    graph.add_node(
        "feed_forward_1",
        OpType::Linear3D,
        {
            normalized1,
            ids.ff1_weight,
            ids.ff1_bias
        },
        {hidden}
    );

    const TensorId activated =
        add_intermediate(
            "ff_activated",
            {
                BATCH,
                TOKENS,
                FF_DIM
            }
        );

    graph.add_node(
        "feed_forward_gelu",
        OpType::GELU,
        {hidden},
        {activated}
    );

    const TensorId feed_forward =
        add_intermediate(
            "feed_forward",
            model_shape
        );

    graph.add_node(
        "feed_forward_2",
        OpType::Linear3D,
        {
            activated,
            ids.ff2_weight,
            ids.ff2_bias
        },
        {feed_forward}
    );

    const TensorId residual2 =
        add_intermediate(
            "residual2",
            model_shape
        );

    graph.add_node(
        "feed_forward_residual",
        OpType::Add,
        {
            normalized1,
            feed_forward
        },
        {residual2}
    );

    ids.output =
        add_intermediate(
            "output",
            model_shape
        );

    graph.add_node(
        "layernorm2",
        OpType::LayerNorm,
        {
            residual2,
            ids.norm2_gamma,
            ids.norm2_beta
        },
        {ids.output}
    );

    graph.mark_output(
        ids.output
    );

    graph.validate();

    return ids;
}


void load_case(
    vortex::GraphExecutor& executor,
    const TransformerIds& ids,
    const std::filesystem::path& directory
) {
    const std::vector<
        std::pair<
            vortex::TensorId,
            std::string
        >
    > files{
        {
            ids.input,
            "input.bin"
        },
        {
            ids.q_weight,
            "q_weight.bin"
        },
        {
            ids.q_bias,
            "q_bias.bin"
        },
        {
            ids.k_weight,
            "k_weight.bin"
        },
        {
            ids.k_bias,
            "k_bias.bin"
        },
        {
            ids.v_weight,
            "v_weight.bin"
        },
        {
            ids.v_bias,
            "v_bias.bin"
        },
        {
            ids.out_weight,
            "out_weight.bin"
        },
        {
            ids.out_bias,
            "out_bias.bin"
        },
        {
            ids.norm1_gamma,
            "norm1_gamma.bin"
        },
        {
            ids.norm1_beta,
            "norm1_beta.bin"
        },
        {
            ids.ff1_weight,
            "ff1_weight.bin"
        },
        {
            ids.ff1_bias,
            "ff1_bias.bin"
        },
        {
            ids.ff2_weight,
            "ff2_weight.bin"
        },
        {
            ids.ff2_bias,
            "ff2_bias.bin"
        },
        {
            ids.norm2_gamma,
            "norm2_gamma.bin"
        },
        {
            ids.norm2_beta,
            "norm2_beta.bin"
        }
    };

    for (
        const auto& [
            tensor_id,
            filename
        ] :
        files
    ) {
        load_tensor(
            executor,
            tensor_id,
            directory / filename
        );
    }
}


struct ErrorMetrics {
    float maximum{};
    double mean{};
};


ErrorMetrics compare(
    const std::vector<float>& result,
    const std::vector<float>& reference
) {
    if (
        result.size() !=
        reference.size()
    ) {
        throw std::runtime_error(
            "Output/reference size mismatch"
        );
    }

    ErrorMetrics metrics;

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {
        const float error =
            std::fabs(
                result[index] -
                reference[index]
            );

        metrics.maximum =
            std::max(
                metrics.maximum,
                error
            );

        metrics.mean +=
            error;
    }

    metrics.mean /=
        static_cast<double>(
            result.size()
        );

    return metrics;
}

} // namespace


int main() {
    using namespace vortex;

    try {
        const std::filesystem::path root =
            "../models/"
            "phase25_transformer_cases";

        const auto cases =
            read_manifest(
                root /
                "manifest.tsv"
            );

        Graph graph;

        const TransformerIds ids =
            build_graph(
                graph
            );

        GraphExecutor executor(
            graph
        );

        float global_max_error =
            0.0f;

        double global_mean_error =
            0.0;

        float global_repeat_error =
            0.0f;

        std::string worst_case;

        std::cout
            << "=================================\n"
            << " VORTEX Randomized Transformer Validation\n"
            << "=================================\n\n";

        std::cout
            << std::scientific
            << std::setprecision(6);

        for (
            const CaseSpec& current :
            cases
        ) {
            const auto directory =
                root /
                current.name;

            load_case(
                executor,
                ids,
                directory
            );

            const auto reference =
                read_fp32(
                    directory /
                    "reference.bin"
                );

            executor.run();

            const Tensor& output_tensor =
                executor.tensor(
                    ids.output
                );

            if (
                !numerics::all_finite(
                    output_tensor
                )
            ) {
                throw std::runtime_error(
                    current.name +
                    " produced NaN or Inf"
                );
            }

            std::vector<float> result(
                output_tensor.numel()
            );

            output_tensor.copy_to_host(
                result.data(),
                result.size() *
                    sizeof(float)
            );

            const ErrorMetrics numerical =
                compare(
                    result,
                    reference
                );

            const std::vector<float>
                first_result =
                    result;

            float repeat_error =
                0.0f;

            for (
                int repetition = 0;
                repetition <
                    REPEATED_EXECUTIONS;
                ++repetition
            ) {
                executor.run();
            }

            executor.tensor(
                ids.output
            ).copy_to_host(
                result.data(),
                result.size() *
                    sizeof(float)
            );

            repeat_error =
                compare(
                    result,
                    first_result
                ).maximum;

            if (
                numerical.maximum >
                global_max_error
            ) {
                global_max_error =
                    numerical.maximum;

                worst_case =
                    current.name;
            }

            global_mean_error =
                std::max(
                    global_mean_error,
                    numerical.mean
                );

            global_repeat_error =
                std::max(
                    global_repeat_error,
                    repeat_error
                );

            std::cout
                << current.name
                << " seed="
                << current.seed
                << " max="
                << numerical.maximum
                << " mean="
                << numerical.mean
                << " repeat="
                << repeat_error
                << '\n';
        }

        std::cout
            << "\nCases:               "
            << cases.size()
            << '\n';

        std::cout
            << "Executions/case:     "
            << (
                REPEATED_EXECUTIONS +
                1
            )
            << '\n';

        std::cout
            << "Worst case:          "
            << worst_case
            << '\n';

        std::cout
            << "Global max error:    "
            << global_max_error
            << '\n';

        std::cout
            << "Worst mean error:    "
            << global_mean_error
            << '\n';

        std::cout
            << "Max repeat error:    "
            << global_repeat_error
            << '\n';

        const bool passed =
            global_max_error <=
                5e-3f &&
            global_mean_error <=
                5e-4 &&
            global_repeat_error <=
                1e-7f;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Randomized transformer validation FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Randomized transformer validation PASSED\n";

        return 0;
    }
    catch (
        const std::exception& error
    ) {
        std::cerr
            << "[VORTEX][ERROR] "
            << error.what()
            << '\n';

        return 1;
    }
}
