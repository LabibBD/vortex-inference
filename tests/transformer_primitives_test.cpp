#include "vortex/graph.hpp"
#include "vortex/graph_executor.hpp"
#include "vortex/operators/transformer.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t BATCH = 3;
    constexpr std::size_t M = 17;
    constexpr std::size_t K = 13;
    constexpr std::size_t N = 11;

    std::vector<float> a(
        BATCH * M * K
    );

    std::vector<float> b(
        BATCH * N * K
    );

    for (
        std::size_t i = 0;
        i < a.size();
        ++i
    ) {
        a[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 19
                ) - 9
            ) * 0.03f;
    }

    for (
        std::size_t i = 0;
        i < b.size();
        ++i
    ) {
        b[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 17
                ) - 8
            ) * 0.025f;
    }

    // ==================================================
    // CPU reference: A * B^T
    // ==================================================

    std::vector<float> reference(
        BATCH * M * N,
        0.0f
    );

    for (
        std::size_t batch = 0;
        batch < BATCH;
        ++batch
    ) {
        for (
            std::size_t row = 0;
            row < M;
            ++row
        ) {
            for (
                std::size_t column = 0;
                column < N;
                ++column
            ) {
                float sum =
                    0.0f;

                for (
                    std::size_t inner = 0;
                    inner < K;
                    ++inner
                ) {
                    sum +=
                        a[
                            batch * M * K +
                            row * K +
                            inner
                        ] *
                        b[
                            batch * N * K +
                            column * K +
                            inner
                        ];
                }

                reference[
                    batch * M * N +
                    row * N +
                    column
                ] =
                    sum;
            }
        }
    }

    // ==================================================
    // Graph
    // ==================================================

    Graph graph;

    const TensorId a_id =
        graph.add_tensor(
            "a",
            {BATCH, M, K},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId b_id =
        graph.add_tensor(
            "b",
            {BATCH, N, K},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Input
        );

    const TensorId b_transposed_id =
        graph.add_tensor(
            "b_transposed",
            {BATCH, K, N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "transpose_b",
        OpType::Transpose,
        {b_id},
        {b_transposed_id}
    );

    const TensorId output_id =
        graph.add_tensor(
            "scores",
            {BATCH, M, N},
            DataType::Float32,
            Device::CUDA,
            TensorRole::Intermediate
        );

    graph.add_node(
        "batched_matmul",
        OpType::BatchedMatMul,
        {
            a_id,
            b_transposed_id
        },
        {output_id}
    );

    graph.mark_output(
        output_id
    );

    GraphExecutor executor(
        graph
    );

    executor.tensor(
        a_id
    ).copy_from_host(
        a.data(),
        a.size() * sizeof(float)
    );

    executor.tensor(
        b_id
    ).copy_from_host(
        b.data(),
        b.size() * sizeof(float)
    );

    executor.run();

    std::vector<float> output(
        reference.size()
    );

    executor.tensor(
        output_id
    ).copy_to_host(
        output.data(),
        output.size() * sizeof(float)
    );

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

    // ==================================================
    // Direct scale test
    // ==================================================

    Tensor scaled(
        {BATCH, M, N},
        DataType::Float32,
        Device::CUDA
    );

    constexpr float SCALE =
        0.25f;

    ops::scale(
        executor.tensor(output_id),
        SCALE,
        scaled
    );

    std::vector<float> scaled_host(
        output.size()
    );

    scaled.copy_to_host(
        scaled_host.data(),
        scaled_host.size() *
            sizeof(float)
    );

    float scale_error =
        0.0f;

    for (
        std::size_t i = 0;
        i < scaled_host.size();
        ++i
    ) {
        scale_error =
            std::max(
                scale_error,
                std::fabs(
                    scaled_host[i] -
                    reference[i] *
                    SCALE
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX Transformer Primitives\n"
        << "=================================\n\n";

    std::cout
        << "Batch:             "
        << BATCH << '\n';

    std::cout
        << "A:                 "
        << M << " x " << K
        << '\n';

    std::cout
        << "B:                 "
        << N << " x " << K
        << '\n';

    std::cout
        << "Output:            "
        << M << " x " << N
        << '\n';

    std::cout
        << "Maximum GEMM error:"
        << " "
        << max_error
        << '\n';

    std::cout
        << "Mean GEMM error:   "
        << mean_error
        << '\n';

    std::cout
        << "Scale error:       "
        << scale_error
        << '\n';

    const bool passed =
        max_error <= 1e-4f &&
        scale_error <= 1e-5f &&
        graph.nodes().size() == 2;

    if (passed) {
        std::cout
            << "\n[VORTEX] Transformer primitives PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Transformer primitives FAILED\n";

    return 1;
}
