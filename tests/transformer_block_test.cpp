#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<float> read_binary(
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

    std::vector<float> result(
        static_cast<std::size_t>(
            bytes
        ) /
        sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(
            result.data()
        ),
        bytes
    );

    if (!file) {
        throw std::runtime_error(
            "Failed reading " +
            path.string()
        );
    }

    return result;
}

void load_tensor(
    vortex::GraphExecutor& executor,
    vortex::TensorId id,
    const std::filesystem::path& path
) {
    const auto data =
        read_binary(path);

    vortex::Tensor& tensor =
        executor.tensor(id);

    if (
        tensor.bytes() !=
        data.size() *
        sizeof(float)
    ) {
        throw std::runtime_error(
            "Tensor/file size mismatch: " +
            path.string()
        );
    }

    tensor.copy_from_host(
        data.data(),
        data.size() *
            sizeof(float)
    );
}

} // namespace

int main() {
    using namespace vortex;

    try {
        constexpr std::size_t BATCH = 4;
        constexpr std::size_t TOKENS = 16;
        constexpr std::size_t MODEL_DIM = 32;
        constexpr std::size_t FF_DIM = 64;

        const std::filesystem::path root =
            "../models/phase21_transformer";

        Graph graph;

        auto add_constant =
            [&](const std::string& name,
                const std::vector<std::size_t>& shape) {
                return graph.add_tensor(
                    name,
                    shape,
                    DataType::Float32,
                    Device::CUDA,
                    TensorRole::Constant
                );
            };

        auto add_intermediate =
            [&](const std::string& name,
                const std::vector<std::size_t>& shape) {
                return graph.add_tensor(
                    name,
                    shape,
                    DataType::Float32,
                    Device::CUDA,
                    TensorRole::Intermediate
                );
            };

        const TensorId input =
            graph.add_tensor(
                "input",
                {
                    BATCH,
                    TOKENS,
                    MODEL_DIM
                },
                DataType::Float32,
                Device::CUDA,
                TensorRole::Input
            );

        const TensorId q_weight =
            add_constant(
                "q.weight",
                {
                    MODEL_DIM,
                    MODEL_DIM
                }
            );

        const TensorId q_bias =
            add_constant(
                "q.bias",
                {MODEL_DIM}
            );

        const TensorId k_weight =
            add_constant(
                "k.weight",
                {
                    MODEL_DIM,
                    MODEL_DIM
                }
            );

        const TensorId k_bias =
            add_constant(
                "k.bias",
                {MODEL_DIM}
            );

        const TensorId v_weight =
            add_constant(
                "v.weight",
                {
                    MODEL_DIM,
                    MODEL_DIM
                }
            );

        const TensorId v_bias =
            add_constant(
                "v.bias",
                {MODEL_DIM}
            );

        const TensorId out_weight =
            add_constant(
                "out.weight",
                {
                    MODEL_DIM,
                    MODEL_DIM
                }
            );

        const TensorId out_bias =
            add_constant(
                "out.bias",
                {MODEL_DIM}
            );

        const TensorId norm1_gamma =
            add_constant(
                "norm1.gamma",
                {MODEL_DIM}
            );

        const TensorId norm1_beta =
            add_constant(
                "norm1.beta",
                {MODEL_DIM}
            );

        const TensorId ff1_weight =
            add_constant(
                "ff1.weight",
                {
                    MODEL_DIM,
                    FF_DIM
                }
            );

        const TensorId ff1_bias =
            add_constant(
                "ff1.bias",
                {FF_DIM}
            );

        const TensorId ff2_weight =
            add_constant(
                "ff2.weight",
                {
                    FF_DIM,
                    MODEL_DIM
                }
            );

        const TensorId ff2_bias =
            add_constant(
                "ff2.bias",
                {MODEL_DIM}
            );

        const TensorId norm2_gamma =
            add_constant(
                "norm2.gamma",
                {MODEL_DIM}
            );

        const TensorId norm2_beta =
            add_constant(
                "norm2.beta",
                {MODEL_DIM}
            );

        const auto model_shape =
            std::vector<std::size_t>{
                BATCH,
                TOKENS,
                MODEL_DIM
            };

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
                input,
                q_weight,
                q_bias
            },
            {query}
        );

        graph.add_node(
            "key_projection",
            OpType::Linear3D,
            {
                input,
                k_weight,
                k_bias
            },
            {key}
        );

        graph.add_node(
            "value_projection",
            OpType::Linear3D,
            {
                input,
                v_weight,
                v_bias
            },
            {value}
        );

        const TensorId context =
            add_intermediate(
                "attention_context",
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
                out_weight,
                out_bias
            },
            {attention_output}
        );

        const TensorId residual1 =
            add_intermediate(
                "attention_residual",
                model_shape
            );

        graph.add_node(
            "attention_residual_add",
            OpType::Add,
            {
                input,
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
                norm1_gamma,
                norm1_beta
            },
            {normalized1}
        );

        const TensorId ff_hidden =
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
                ff1_weight,
                ff1_bias
            },
            {ff_hidden}
        );

        const TensorId ff_activated =
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
            {ff_hidden},
            {ff_activated}
        );

        const TensorId ff_output =
            add_intermediate(
                "ff_output",
                model_shape
            );

        graph.add_node(
            "feed_forward_2",
            OpType::Linear3D,
            {
                ff_activated,
                ff2_weight,
                ff2_bias
            },
            {ff_output}
        );

        const TensorId residual2 =
            add_intermediate(
                "feed_forward_residual",
                model_shape
            );

        graph.add_node(
            "feed_forward_residual_add",
            OpType::Add,
            {
                normalized1,
                ff_output
            },
            {residual2}
        );

        const TensorId output =
            add_intermediate(
                "output",
                model_shape
            );

        graph.add_node(
            "layernorm2",
            OpType::LayerNorm,
            {
                residual2,
                norm2_gamma,
                norm2_beta
            },
            {output}
        );

        graph.mark_output(output);

        GraphExecutor executor(graph);

        load_tensor(
            executor,
            input,
            root / "input.bin"
        );

        const std::vector<
            std::pair<
                TensorId,
                std::string
            >
        > constant_files = {
            {q_weight, "q_weight.bin"},
            {q_bias, "q_bias.bin"},
            {k_weight, "k_weight.bin"},
            {k_bias, "k_bias.bin"},
            {v_weight, "v_weight.bin"},
            {v_bias, "v_bias.bin"},
            {out_weight, "out_weight.bin"},
            {out_bias, "out_bias.bin"},
            {norm1_gamma, "norm1_gamma.bin"},
            {norm1_beta, "norm1_beta.bin"},
            {ff1_weight, "ff1_weight.bin"},
            {ff1_bias, "ff1_bias.bin"},
            {ff2_weight, "ff2_weight.bin"},
            {ff2_bias, "ff2_bias.bin"},
            {norm2_gamma, "norm2_gamma.bin"},
            {norm2_beta, "norm2_beta.bin"},
        };

        for (
            const auto& [
                tensor_id,
                filename
            ] :
            constant_files
        ) {
            load_tensor(
                executor,
                tensor_id,
                root / filename
            );
        }

        executor.run();

        const auto reference =
            read_binary(
                root /
                "reference.bin"
            );

        std::vector<float> result(
            executor.tensor(
                output
            ).numel()
        );

        executor.tensor(
            output
        ).copy_to_host(
            result.data(),
            result.size() *
                sizeof(float)
        );

        if (
            result.size() !=
            reference.size()
        ) {
            throw std::runtime_error(
                "Transformer reference size mismatch"
            );
        }

        float max_error = 0.0f;
        double mean_error = 0.0;
        bool finite = true;

        for (
            std::size_t i = 0;
            i < result.size();
            ++i
        ) {
            finite =
                finite &&
                std::isfinite(
                    result[i]
                );

            const float error =
                std::fabs(
                    result[i] -
                    reference[i]
                );

            max_error =
                std::max(
                    max_error,
                    error
                );

            mean_error += error;
        }

        mean_error /=
            static_cast<double>(
                result.size()
            );

        constexpr int WARMUP = 10;
        constexpr int ITERATIONS = 50;

        for (
            int i = 0;
            i < WARMUP;
            ++i
        ) {
            executor.run();
        }

        std::vector<double> samples;
        samples.reserve(ITERATIONS);

        for (
            int i = 0;
            i < ITERATIONS;
            ++i
        ) {
            const auto start =
                std::chrono::
                    steady_clock::now();

            executor.run();

            const auto stop =
                std::chrono::
                    steady_clock::now();

            samples.push_back(
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    stop - start
                ).count()
            );
        }

        const double mean_latency =
            std::accumulate(
                samples.begin(),
                samples.end(),
                0.0
            ) /
            samples.size();

        std::sort(
            samples.begin(),
            samples.end()
        );

        const double median_latency =
            samples[
                samples.size() / 2
            ];

        const double p95_latency =
            samples[
                static_cast<std::size_t>(
                    0.95 *
                    (
                        samples.size() -
                        1
                    )
                )
            ];

        std::cout
            << "=================================\n"
            << " VORTEX Transformer Block\n"
            << "=================================\n\n";

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "Batch:              "
            << BATCH << '\n';

        std::cout
            << "Tokens:             "
            << TOKENS << '\n';

        std::cout
            << "Model dimension:    "
            << MODEL_DIM << '\n';

        std::cout
            << "FFN dimension:      "
            << FF_DIM << '\n';

        std::cout
            << "Graph nodes:        "
            << graph.nodes().size()
            << "\n\n";

        std::cout
            << "Mean latency:       "
            << mean_latency
            << " ms\n";

        std::cout
            << "Median latency:     "
            << median_latency
            << " ms\n";

        std::cout
            << "P95 latency:        "
            << p95_latency
            << " ms\n\n";

        std::cout
            << "Maximum abs error:  "
            << max_error << '\n';

        std::cout
            << "Mean abs error:     "
            << mean_error << '\n';

        std::cout
            << "All values finite:  "
            << (
                finite
                    ? "YES"
                    : "NO"
            )
            << '\n';

        const bool passed =
            finite &&
            max_error <= 2e-3f &&
            mean_error <= 2e-4;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Transformer block FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Transformer block PASSED\n";

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
