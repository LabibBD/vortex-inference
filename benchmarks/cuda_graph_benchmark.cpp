#include "vortex/cuda_graph.hpp"
#include "vortex/cuda_stream.hpp"
#include "vortex/operators/activations.hpp"
#include "vortex/tensor.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock =
    std::chrono::steady_clock;

double elapsed_ms(
    Clock::time_point start,
    Clock::time_point stop
) {
    return std::chrono::duration<
        double,
        std::milli
    >(stop - start).count();
}

} // namespace

int main() {
    using namespace vortex;

    constexpr std::size_t N =
        4096;

    constexpr int KERNELS =
        20;

    constexpr int WARMUP =
        50;

    constexpr int ITERATIONS =
        500;

    std::vector<float> host(
        N,
        0.5f
    );

    Tensor input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor buffer_a(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor buffer_b(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    input.copy_from_host(
        host.data(),
        host.size() *
            sizeof(float)
    );

    CudaStream stream;

    auto launch_sequence =
        [&](cudaStream_t cuda_stream) {
            const Tensor* source =
                &input;

            Tensor* destination =
                &buffer_a;

            for (
                int kernel = 0;
                kernel < KERNELS;
                ++kernel
            ) {
                if (kernel % 2 == 0) {
                    ops::gelu_async(
                        *source,
                        *destination,
                        cuda_stream
                    );
                } else {
                    ops::relu_async(
                        *source,
                        *destination,
                        cuda_stream
                    );
                }

                source =
                    destination;

                destination =
                    (
                        destination ==
                        &buffer_a
                    )
                        ? &buffer_b
                        : &buffer_a;
            }
        };

    // ==================================================
    // Warm up normal path.
    // ==================================================

    for (int i = 0; i < WARMUP; ++i) {
        launch_sequence(
            stream.get()
        );

        stream.synchronize();
    }

    // ==================================================
    // Capture sequence.
    // ==================================================

    CudaGraph graph;

    graph.capture(
        stream,
        launch_sequence
    );

    // Warm up graph replay.
    for (int i = 0; i < WARMUP; ++i) {
        graph.launch(
            stream.get()
        );

        stream.synchronize();
    }

    // ==================================================
    // Normal launches
    // ==================================================

    double normal_total =
        0.0;

    for (
        int i = 0;
        i < ITERATIONS;
        ++i
    ) {
        const auto start =
            Clock::now();

        launch_sequence(
            stream.get()
        );

        stream.synchronize();

        const auto stop =
            Clock::now();

        normal_total +=
            elapsed_ms(
                start,
                stop
            );
    }

    // ==================================================
    // CUDA Graph replay
    // ==================================================

    double graph_total =
        0.0;

    for (
        int i = 0;
        i < ITERATIONS;
        ++i
    ) {
        const auto start =
            Clock::now();

        graph.launch(
            stream.get()
        );

        stream.synchronize();

        const auto stop =
            Clock::now();

        graph_total +=
            elapsed_ms(
                start,
                stop
            );
    }

    const double normal_ms =
        normal_total /
        ITERATIONS;

    const double graph_ms =
        graph_total /
        ITERATIONS;

    const double speedup =
        normal_ms /
        graph_ms;

    std::cout
        << "=================================\n"
        << " VORTEX CUDA Graph Benchmark\n"
        << "=================================\n\n";

    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "Elements:          "
        << N << '\n';

    std::cout
        << "Kernels/sequence:  "
        << KERNELS << '\n';

    std::cout
        << "Normal execution:  "
        << normal_ms
        << " ms\n";

    std::cout
        << "CUDA Graph replay: "
        << graph_ms
        << " ms\n";

    std::cout
        << "Speedup:           "
        << speedup
        << "x\n";

    return 0;
}
