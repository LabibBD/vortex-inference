#include "vortex/graph_executor.hpp"
#include "vortex/model.hpp"
#include "vortex/numerics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model_directory;
    std::filesystem::path input_path;
    std::filesystem::path labels_path;

    std::optional<
        std::filesystem::path
    > reference_path;

    float tolerance =
        1e-4f;

    bool json =
        false;
};


std::string json_escape(
    const std::string& value
) {
    std::string result;

    for (const char character : value) {
        switch (character) {
            case '"':
                result += "\\\"";
                break;

            case '\\':
                result += "\\\\";
                break;

            case '\n':
                result += "\\n";
                break;

            case '\r':
                result += "\\r";
                break;

            case '\t':
                result += "\\t";
                break;

            default:
                result += character;
                break;
        }
    }

    return result;
}


std::vector<float> read_fp32(
    const std::filesystem::path& path
) {
    std::ifstream file(
        path,
        std::ios::binary |
        std::ios::ate
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open FP32 file: " +
            path.string()
        );
    }

    const std::streamoff byte_count =
        file.tellg();

    if (
        byte_count < 0 ||
        byte_count %
            static_cast<std::streamoff>(
                sizeof(float)
            ) != 0
    ) {
        throw std::runtime_error(
            "Invalid FP32 file size: " +
            path.string()
        );
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<float> values(
        static_cast<std::size_t>(
            byte_count
        ) /
        sizeof(float)
    );

    if (!values.empty()) {
        file.read(
            reinterpret_cast<char*>(
                values.data()
            ),
            static_cast<std::streamsize>(
                byte_count
            )
        );
    }

    if (!file) {
        throw std::runtime_error(
            "Failed reading FP32 file: " +
            path.string()
        );
    }

    return values;
}


std::vector<std::string> read_labels(
    const std::filesystem::path& path
) {
    std::ifstream file(
        path
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open labels file: " +
            path.string()
        );
    }

    std::vector<std::string> labels;

    std::string line;

    while (
        std::getline(
            file,
            line
        )
    ) {
        if (!line.empty()) {
            labels.push_back(
                line
            );
        }
    }

    if (labels.empty()) {
        throw std::runtime_error(
            "Labels file contains no labels"
        );
    }

    return labels;
}


Options parse_arguments(
    int argc,
    char** argv
) {
    Options options;

    const auto require_value =
        [&](
            int& index,
            const std::string& argument
        ) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    argument +
                    " requires a value"
                );
            }

            ++index;

            return argv[index];
        };

    for (
        int index = 1;
        index < argc;
        ++index
    ) {
        const std::string argument =
            argv[index];

        if (argument == "--model") {
            options.model_directory =
                require_value(
                    index,
                    argument
                );
        }
        else if (argument == "--input") {
            options.input_path =
                require_value(
                    index,
                    argument
                );
        }
        else if (argument == "--labels") {
            options.labels_path =
                require_value(
                    index,
                    argument
                );
        }
        else if (argument == "--reference") {
            options.reference_path =
                require_value(
                    index,
                    argument
                );
        }
        else if (argument == "--tolerance") {
            options.tolerance =
                std::stof(
                    require_value(
                        index,
                        argument
                    )
                );
        }
        else if (argument == "--json") {
            options.json =
                true;
        }
        else if (
            argument == "--help" ||
            argument == "-h"
        ) {
            std::cout
                << "Usage:\n"
                << "  vortex_infer"
                << " --model MODEL_DIRECTORY"
                << " --input INPUT.bin"
                << " --labels LABELS.txt"
                << " [--reference REFERENCE.bin]"
                << " [--tolerance VALUE]"
                << " [--json]\n";

            std::exit(0);
        }
        else {
            throw std::invalid_argument(
                "Unknown argument: " +
                argument
            );
        }
    }

    if (options.model_directory.empty()) {
        throw std::invalid_argument(
            "--model is required"
        );
    }

    if (options.input_path.empty()) {
        throw std::invalid_argument(
            "--input is required"
        );
    }

    if (options.labels_path.empty()) {
        throw std::invalid_argument(
            "--labels is required"
        );
    }

    if (
        !std::isfinite(
            options.tolerance
        ) ||
        options.tolerance < 0.0f
    ) {
        throw std::invalid_argument(
            "Tolerance must be finite and non-negative"
        );
    }

    return options;
}


void print_json(
    const std::vector<std::string>& labels,
    const std::vector<float>& probabilities,
    std::size_t prediction,
    double latency_ms,
    const std::optional<float>& max_error,
    float tolerance,
    bool validation_passed
) {
    std::cout
        << std::fixed
        << std::setprecision(8);

    std::cout
        << "{\n";

    std::cout
        << "  \"prediction\": \""
        << json_escape(
            labels[prediction]
        )
        << "\",\n";

    std::cout
        << "  \"class_index\": "
        << prediction
        << ",\n";

    std::cout
        << "  \"confidence\": "
        << probabilities[prediction]
        << ",\n";

    std::cout
        << "  \"latency_ms\": "
        << latency_ms
        << ",\n";

    std::cout
        << "  \"backend\": "
        << "\"VORTEX CUDA\",\n";

    std::cout
        << "  \"probabilities\": {\n";

    for (
        std::size_t index = 0;
        index < labels.size();
        ++index
    ) {
        std::cout
            << "    \""
            << json_escape(
                labels[index]
            )
            << "\": "
            << probabilities[index];

        if (
            index + 1 <
            labels.size()
        ) {
            std::cout
                << ',';
        }

        std::cout
            << '\n';
    }

    std::cout
        << "  }";

    if (max_error.has_value()) {
        std::cout
            << ",\n"
            << "  \"validation\": {\n"
            << "    \"max_abs_error\": "
            << *max_error
            << ",\n"
            << "    \"tolerance\": "
            << tolerance
            << ",\n"
            << "    \"passed\": "
            << (
                validation_passed
                    ? "true"
                    : "false"
            )
            << "\n"
            << "  }\n";
    }
    else {
        std::cout
            << '\n';
    }

    std::cout
        << "}\n";
}


void print_human(
    const std::vector<std::string>& labels,
    const std::vector<float>& probabilities,
    std::size_t prediction,
    double latency_ms,
    const std::optional<float>& max_error,
    float tolerance,
    bool validation_passed
) {
    std::cout
        << std::fixed
        << std::setprecision(8);

    std::cout
        << "=================================\n"
        << " VORTEX Inference\n"
        << "=================================\n\n";

    for (
        std::size_t index = 0;
        index < labels.size();
        ++index
    ) {
        std::cout
            << std::left
            << std::setw(12)
            << labels[index]
            << probabilities[index]
            << '\n';
    }

    std::cout
        << "\nPrediction: "
        << labels[prediction]
        << '\n';

    std::cout
        << "Confidence: "
        << probabilities[prediction]
        << '\n';

    std::cout
        << "Latency:    "
        << latency_ms
        << " ms\n";

    if (max_error.has_value()) {
        std::cout
            << "Max error:  "
            << *max_error
            << '\n';

        std::cout
            << "Tolerance:  "
            << tolerance
            << '\n';

        std::cout
            << "Validation: "
            << (
                validation_passed
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';
    }
}

} // namespace


int main(
    int argc,
    char** argv
) {
    try {
        const Options options =
            parse_arguments(
                argc,
                argv
            );

        vortex::Model model =
            vortex::ModelLoader::load(
                options.model_directory
            );

        if (
            model.inputs().size() != 1 ||
            model.outputs().size() != 1
        ) {
            throw std::runtime_error(
                "vortex_infer currently requires exactly "
                "one input and one output"
            );
        }

        vortex::GraphExecutor executor(
            model.graph()
        );

        model.load_constants(
            executor
        );

        vortex::Tensor& input_tensor =
            executor.tensor(
                model.inputs().front()
            );

        const std::vector<float>
            input_values =
                read_fp32(
                    options.input_path
                );

        if (
            input_values.size() *
                sizeof(float) !=
            input_tensor.bytes()
        ) {
            throw std::runtime_error(
                "Input file size does not match "
                "the model input tensor"
            );
        }

        input_tensor.copy_from_host(
            input_values.data(),
            input_values.size() *
                sizeof(float)
        );

        const auto start =
            std::chrono::
                steady_clock::now();

        executor.run();

        const auto stop =
            std::chrono::
                steady_clock::now();

        const double latency_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                stop - start
            ).count();

        const vortex::Tensor&
            output_tensor =
                executor.tensor(
                    model.outputs().front()
                );

        if (
            !vortex::numerics::all_finite(
                output_tensor
            )
        ) {
            throw std::runtime_error(
                "Model produced NaN or infinity"
            );
        }

        std::vector<float> probabilities(
            output_tensor.numel()
        );

        output_tensor.copy_to_host(
            probabilities.data(),
            probabilities.size() *
                sizeof(float)
        );

        const std::vector<std::string>
            labels =
                read_labels(
                    options.labels_path
                );

        if (
            labels.size() !=
            probabilities.size()
        ) {
            throw std::runtime_error(
                "Number of labels does not match "
                "the output tensor size"
            );
        }

        const auto prediction_iterator =
            std::max_element(
                probabilities.begin(),
                probabilities.end()
            );

        const std::size_t prediction =
            static_cast<std::size_t>(
                std::distance(
                    probabilities.begin(),
                    prediction_iterator
                )
            );

        std::optional<float> max_error;

        bool validation_passed =
            true;

        if (
            options.reference_path
                .has_value()
        ) {
            const std::vector<float>
                reference =
                    read_fp32(
                        *options.reference_path
                    );

            if (
                reference.size() !=
                probabilities.size()
            ) {
                throw std::runtime_error(
                    "Reference output size mismatch"
                );
            }

            float measured_max_error =
                0.0f;

            for (
                std::size_t index = 0;
                index < reference.size();
                ++index
            ) {
                measured_max_error =
                    std::max(
                        measured_max_error,
                        std::fabs(
                            probabilities[index] -
                            reference[index]
                        )
                    );
            }

            max_error =
                measured_max_error;

            validation_passed =
                measured_max_error <=
                options.tolerance;
        }

        if (options.json) {
            print_json(
                labels,
                probabilities,
                prediction,
                latency_ms,
                max_error,
                options.tolerance,
                validation_passed
            );
        }
        else {
            print_human(
                labels,
                probabilities,
                prediction,
                latency_ms,
                max_error,
                options.tolerance,
                validation_passed
            );
        }

        return validation_passed
            ? 0
            : 2;
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
