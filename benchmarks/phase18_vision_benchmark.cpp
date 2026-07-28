#include "vortex/graph_executor.hpp"
#include "vortex/memory_pool.hpp"
#include "vortex/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using Clock =
    std::chrono::steady_clock;

template <typename T>
std::vector<T> read_binary(
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

    const auto byte_count =
        file.tellg();

    if (
        byte_count <= 0 ||
        byte_count %
            static_cast<std::streamoff>(
                sizeof(T)
            ) != 0
    ) {
        throw std::runtime_error(
            "Invalid binary data file"
        );
    }

    file.seekg(
        0,
        std::ios::beg
    );

    std::vector<T> result(
        static_cast<std::size_t>(
            byte_count
        ) /
        sizeof(T)
    );

    file.read(
        reinterpret_cast<char*>(
            result.data()
        ),
        byte_count
    );

    if (!file) {
        throw std::runtime_error(
            "Failed reading binary file"
        );
    }

    return result;
}

std::size_t argmax(
    const float* data,
    std::size_t count
) {
    return static_cast<std::size_t>(
        std::max_element(
            data,
            data + count
        ) -
        data
    );
}

double percentile(
    std::vector<double> data,
    double fraction
) {
    std::sort(
        data.begin(),
        data.end()
    );

    const std::size_t index =
        static_cast<std::size_t>(
            fraction *
            static_cast<double>(
                data.size() - 1
            )
        );

    return data[index];
}

} // namespace

int main() {
    using namespace vortex;

    try {
        constexpr std::size_t BATCH =
            1024;

        constexpr std::size_t CLASSES =
            10;

        constexpr int WARMUP =
            30;

        constexpr int ITERATIONS =
            200;

        auto& pool =
            MemoryPool::global();

        pool.clear();

        Model model =
            ModelLoader::load(
                "../models/"
                "phase18_mnist.vortex"
            );

        GraphExecutor executor(
            model.graph()
        );

        model.load_constants(
            executor
        );

        const auto input =
            read_binary<float>(
                "../models/"
                "phase18_input.bin"
            );

        const auto reference =
            read_binary<float>(
                "../models/"
                "phase18_reference.bin"
            );

        const auto labels =
            read_binary<std::int64_t>(
                "../models/"
                "phase18_labels.bin"
            );

        if (
            model.inputs().size() != 1 ||
            model.outputs().size() != 1
        ) {
            throw std::runtime_error(
                "Vision model must have one "
                "input and one output"
            );
        }

        const TensorId input_id =
            model.inputs()[0];

        const TensorId output_id =
            model.outputs()[0];

        Tensor& input_tensor =
            executor.tensor(
                input_id
            );

        if (
            input_tensor.bytes() !=
            input.size() *
            sizeof(float)
        ) {
            throw std::runtime_error(
                "MNIST input size mismatch"
            );
        }

        input_tensor.copy_from_host(
            input.data(),
            input.size() *
                sizeof(float)
        );

        // ==============================================
        // Warmup
        // ==============================================

        for (
            int i = 0;
            i < WARMUP;
            ++i
        ) {
            executor.run();
        }

        // ==============================================
        // Benchmark
        // ==============================================

        std::vector<double> samples;

        samples.reserve(
            ITERATIONS
        );

        for (
            int i = 0;
            i < ITERATIONS;
            ++i
        ) {
            const auto start =
                Clock::now();

            executor.run();

            const auto stop =
                Clock::now();

            samples.push_back(
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    stop - start
                ).count()
            );
        }

        const double mean_ms =
            std::accumulate(
                samples.begin(),
                samples.end(),
                0.0
            ) /
            samples.size();

        const double median_ms =
            percentile(
                samples,
                0.50
            );

        const double p95_ms =
            percentile(
                samples,
                0.95
            );

        const Tensor& output_tensor =
            executor.tensor(
                output_id
            );

        std::vector<float> output(
            output_tensor.numel()
        );

        output_tensor.copy_to_host(
            output.data(),
            output.size() *
                sizeof(float)
        );

        if (
            output.size() !=
            reference.size() ||
            labels.size() != BATCH
        ) {
            throw std::runtime_error(
                "Vision validation size mismatch"
            );
        }

        // ==============================================
        // Numerical comparison
        // ==============================================

        float max_error =
            0.0f;

        double mean_error =
            0.0;

        for (
            std::size_t i = 0;
            i < output.size();
            ++i
        ) {
            const float error =
                std::fabs(
                    output[i] -
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
            output.size();

        // ==============================================
        // Real classification accuracy
        // ==============================================

        std::size_t vortex_correct =
            0;

        std::size_t reference_correct =
            0;

        std::size_t agreement =
            0;

        for (
            std::size_t row = 0;
            row < BATCH;
            ++row
        ) {
            const auto vortex_class =
                argmax(
                    output.data() +
                    row * CLASSES,
                    CLASSES
                );

            const auto reference_class =
                argmax(
                    reference.data() +
                    row * CLASSES,
                    CLASSES
                );

            const auto label =
                static_cast<std::size_t>(
                    labels[row]
                );

            if (
                vortex_class == label
            ) {
                ++vortex_correct;
            }

            if (
                reference_class == label
            ) {
                ++reference_correct;
            }

            if (
                vortex_class ==
                reference_class
            ) {
                ++agreement;
            }
        }

        const double vortex_accuracy =
            100.0 *
            vortex_correct /
            static_cast<double>(
                BATCH
            );

        const double pytorch_accuracy =
            100.0 *
            reference_correct /
            static_cast<double>(
                BATCH
            );

        const double agreement_percent =
            100.0 *
            agreement /
            static_cast<double>(
                BATCH
            );

        const double throughput =
            BATCH /
            (mean_ms / 1000.0);

        const auto memory =
            pool.stats();

        const double peak_mib =
            memory.peak_active_bytes /
            (1024.0 * 1024.0);

        std::cout
            << "=================================\n"
            << " VORTEX Phase-18 Vision Benchmark\n"
            << "=================================\n\n";

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "Dataset:             MNIST\n";

        std::cout
            << "Architecture:        "
            << "784 -> 256 -> 128 -> 10\n";

        std::cout
            << "Images:              "
            << BATCH << "\n\n";

        std::cout
            << "Mean latency:        "
            << mean_ms << " ms\n";

        std::cout
            << "Median latency:      "
            << median_ms << " ms\n";

        std::cout
            << "P95 latency:         "
            << p95_ms << " ms\n";

        std::cout
            << "Throughput:          "
            << throughput
            << " images/s\n";

        std::cout
            << "Peak managed VRAM:   "
            << peak_mib
            << " MiB\n\n";

        std::cout
            << "PyTorch accuracy:    "
            << pytorch_accuracy
            << "%\n";

        std::cout
            << "VORTEX accuracy:     "
            << vortex_accuracy
            << "%\n";

        std::cout
            << "Prediction agreement:"
            << " "
            << agreement_percent
            << "%\n\n";

        std::cout
            << "Maximum abs error:   "
            << max_error << '\n';

        std::cout
            << "Mean abs error:      "
            << mean_error << '\n';

        std::filesystem::
            create_directories(
                "../results"
            );

        std::ofstream csv(
            "../results/"
            "phase18_vision.csv"
        );

        csv
            << "dataset,batch,mean_ms,"
            << "median_ms,p95_ms,"
            << "images_per_second,"
            << "peak_managed_mib,"
            << "pytorch_accuracy,"
            << "vortex_accuracy,"
            << "agreement,max_error,"
            << "mean_error\n";

        csv
            << "MNIST,"
            << BATCH << ','
            << mean_ms << ','
            << median_ms << ','
            << p95_ms << ','
            << throughput << ','
            << peak_mib << ','
            << pytorch_accuracy << ','
            << vortex_accuracy << ','
            << agreement_percent << ','
            << max_error << ','
            << mean_error << '\n';

        const bool passed =
            max_error <= 1e-3f &&
            mean_error <= 1e-4 &&
            agreement_percent >= 99.8 &&
            vortex_accuracy >= 90.0;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Vision validation FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Real vision inference PASSED\n";

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
