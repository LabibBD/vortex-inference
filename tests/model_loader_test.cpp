#include "vortex/graph_executor.hpp"
#include "vortex/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> read_float_file(
    const char* path
) {
    std::ifstream file(
        path,
        std::ios::binary |
        std::ios::ate
    );

    if (!file) {
        throw std::runtime_error(
            std::string(
                "Could not open "
            ) +
            path
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
            "Invalid FP32 binary file"
        );
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<float> data(
        static_cast<std::size_t>(
            bytes
        ) /
        sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(
            data.data()
        ),
        bytes
    );

    if (!file) {
        throw std::runtime_error(
            "Failed reading FP32 file"
        );
    }

    return data;
}

} // namespace

int main() {
    using namespace vortex;

    try {
        Model model =
            ModelLoader::load(
                "../models/phase16_mlp.vortex"
            );

        std::cout
            << "=================================\n"
            << " VORTEX Model Loader Test\n"
            << "=================================\n\n";

        std::cout
            << "Graph tensors: "
            << model.graph().tensors().size()
            << '\n';

        std::cout
            << "Graph nodes:   "
            << model.graph().nodes().size()
            << '\n';

        std::cout
            << "Inputs:        "
            << model.inputs().size()
            << '\n';

        std::cout
            << "Outputs:       "
            << model.outputs().size()
            << '\n';

        if (
            model.inputs().size() != 1 ||
            model.outputs().size() != 1
        ) {
            throw std::runtime_error(
                "Phase-16 model must have "
                "one input and one output"
            );
        }

        GraphExecutor executor(
            model.graph()
        );

        // Load converted ONNX weights.
        model.load_constants(
            executor
        );

        const auto input =
            read_float_file(
                "../models/phase16_input.bin"
            );

        const auto reference =
            read_float_file(
                "../models/phase16_reference.bin"
            );

        const TensorId input_id =
            model.inputs()[0];

        const TensorId output_id =
            model.outputs()[0];

        Tensor& input_tensor =
            executor.tensor(
                input_id
            );

        if (
            input.size() *
            sizeof(float) !=
            input_tensor.bytes()
        ) {
            throw std::runtime_error(
                "Reference input size mismatch"
            );
        }

        input_tensor.copy_from_host(
            input.data(),
            input.size() *
                sizeof(float)
        );

        // ==============================================
        // Entire imported model runs through VORTEX.
        // ==============================================

        executor.run();

        const Tensor& output_tensor =
            executor.tensor(
                output_id
            );

        std::vector<float> result(
            output_tensor.numel()
        );

        output_tensor.copy_to_host(
            result.data(),
            result.size() *
                sizeof(float)
        );

        if (
            result.size() !=
            reference.size()
        ) {
            throw std::runtime_error(
                "Reference output size mismatch"
            );
        }

        float max_error =
            0.0f;

        double mean_error =
            0.0;

        for (
            std::size_t i = 0;
            i < result.size();
            ++i
        ) {
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

            mean_error +=
                error;
        }

        mean_error /=
            static_cast<double>(
                result.size()
            );

        std::cout
            << "\nMaximum abs error: "
            << max_error
            << '\n';

        std::cout
            << "Mean abs error:    "
            << mean_error
            << '\n';

        // Softmax sanity check.
        const auto& shape =
            output_tensor.shape();

        float max_row_sum_error =
            0.0f;

        if (shape.size() == 2) {
            const std::size_t rows =
                shape[0];

            const std::size_t columns =
                shape[1];

            for (
                std::size_t row = 0;
                row < rows;
                ++row
            ) {
                float sum =
                    0.0f;

                for (
                    std::size_t col = 0;
                    col < columns;
                    ++col
                ) {
                    sum +=
                        result[
                            row *
                            columns +
                            col
                        ];
                }

                max_row_sum_error =
                    std::max(
                        max_row_sum_error,
                        std::fabs(
                            sum - 1.0f
                        )
                    );
            }
        }

        std::cout
            << "Max row-sum error: "
            << max_row_sum_error
            << '\n';

        const bool passed =
            max_error <= 1e-4f &&
            mean_error <= 1e-5 &&
            max_row_sum_error <= 1e-5f;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Imported model FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Imported model inference PASSED\n";

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
