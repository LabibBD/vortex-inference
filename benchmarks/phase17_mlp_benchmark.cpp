#include "vortex/graph_executor.hpp"
#include "vortex/memory_pool.hpp"
#include "vortex/model.hpp"

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
#include <vector>

namespace {

using Clock =
    std::chrono::steady_clock;

constexpr int WARMUP =
    50;

constexpr int ITERATIONS =
    500;

std::vector<float>
read_float_file(
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
            "Invalid FP32 file"
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
            "Could not read FP32 file"
        );
    }

    return result;
}

double percentile(
    std::vector<double> values,
    double fraction
) {
    std::sort(
        values.begin(),
        values.end()
    );

    const std::size_t index =
        static_cast<std::size_t>(
            fraction *
            static_cast<double>(
                values.size() - 1
            )
        );

    return values[index];
}

std::size_t argmax(
    const float* values,
    std::size_t count
) {
    return static_cast<std::size_t>(
        std::max_element(
            values,
            values + count
        ) -
        values
    );
}

} // namespace

int main() {
    using namespace vortex;

    try {
        constexpr std::size_t BATCH =
            128;

        constexpr std::size_t CLASSES =
            10;

        Model model =
            ModelLoader::load(
                "../models/phase17_mlp.vortex"
            );

        if (
            model.inputs().size() != 1 ||
            model.outputs().size() != 1
        ) {
            throw std::runtime_error(
                "Phase-17 model must have "
                "one input and one output"
            );
        }

        GraphExecutor executor(
            model.graph()
        );

        model.load_constants(
            executor
        );

        const auto input =
            read_float_file(
                "../models/phase17_input.bin"
            );

        const auto reference =
            read_float_file(
                "../models/phase17_reference.bin"
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
            input_tensor.bytes() !=
            input.size() * sizeof(float)
        ) {
            throw std::runtime_error(
                "Phase-17 input size mismatch"
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

        const double mean =
            std::accumulate(
                samples.begin(),
                samples.end(),
                0.0
            ) /
            static_cast<double>(
                samples.size()
            );

        const double median =
            percentile(
                samples,
                0.50
            );

        const double p95 =
            percentile(
                samples,
                0.95
            );

        // ==============================================
        // Numerical validation
        // ==============================================

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
            reference.size()
        ) {
            throw std::runtime_error(
                "Reference/output size mismatch"
            );
        }

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

            mean_error +=
                error;
        }

        mean_error /=
            static_cast<double>(
                output.size()
            );

        // ==============================================
        // Classification agreement
        // ==============================================

        std::size_t matches =
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

            if (
                vortex_class ==
                reference_class
            ) {
                ++matches;
            }
        }

        const double agreement =
            100.0 *
            static_cast<double>(
                matches
            ) /
            static_cast<double>(
                BATCH
            );

        const auto memory =
            MemoryPool::global()
                .stats();

        const double managed_mib =
            static_cast<double>(
                memory.peak_active_bytes
            ) /
            (1024.0 * 1024.0);

        const double throughput =
            (
                static_cast<double>(
                    BATCH
                ) /
                (mean / 1000.0)
            );

        std::cout
            << "=================================\n"
            << " VORTEX Phase-17 MLP Benchmark\n"
            << "=================================\n\n";

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "Architecture:       "
            << "784 -> 512 -> 256 -> 10\n";

        std::cout
            << "Batch size:         "
            << BATCH << '\n';

        std::cout
            << "Warmup:             "
            << WARMUP << '\n';

        std::cout
            << "Iterations:         "
            << ITERATIONS << "\n\n";

        std::cout
            << "Mean latency:       "
            << mean << " ms\n";

        std::cout
            << "Median latency:     "
            << median << " ms\n";

        std::cout
            << "P95 latency:        "
            << p95 << " ms\n";

        std::cout
            << "Throughput:         "
            << throughput
            << " samples/s\n";

        std::cout
            << "Managed peak VRAM:  "
            << managed_mib
            << " MiB\n\n";

        std::cout
            << "Maximum abs error:  "
            << max_error << '\n';

        std::cout
            << "Mean abs error:     "
            << mean_error << '\n';

        std::cout
            << "Argmax agreement:   "
            << agreement << "%\n";

        // ==============================================
        // Save benchmark result
        // ==============================================

        std::filesystem::
            create_directories(
                "../results"
            );

        std::ofstream csv(
            "../results/"
            "phase17_vortex.csv"
        );

        csv
            << "runtime,batch,mean_ms,"
            << "median_ms,p95_ms,"
            << "samples_per_second,"
            << "peak_managed_mib,"
            << "max_error,mean_error,"
            << "argmax_agreement\n";

        csv
            << "VORTEX,"
            << BATCH << ','
            << mean << ','
            << median << ','
            << p95 << ','
            << throughput << ','
            << managed_mib << ','
            << max_error << ','
            << mean_error << ','
            << agreement << '\n';

        const bool passed =
            max_error <= 2e-4f &&
            mean_error <= 2e-5 &&
            matches == BATCH;

        if (!passed) {
            std::cerr
                << "\n[VORTEX] Phase-17 validation FAILED\n";

            return 1;
        }

        std::cout
            << "\n[VORTEX] Phase-17 model PASSED\n";

        std::cout
            << "[VORTEX] Saved "
            << "results/phase17_vortex.csv\n";

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
