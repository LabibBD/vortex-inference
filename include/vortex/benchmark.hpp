#pragma once

#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vortex {

struct BenchmarkStats {
    double mean_ms{0.0};
    double median_ms{0.0};
    double p95_ms{0.0};
    double stddev_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};

    int warmup_iterations{0};
    int measured_iterations{0};
};

inline BenchmarkStats summarize_benchmark(
    const std::vector<double>& samples,
    int warmup_iterations
) {
    if (samples.empty()) {
        throw std::invalid_argument(
            "Benchmark samples cannot be empty"
        );
    }

    BenchmarkStats stats;

    stats.warmup_iterations =
        warmup_iterations;

    stats.measured_iterations =
        static_cast<int>(samples.size());

    stats.mean_ms =
        std::accumulate(
            samples.begin(),
            samples.end(),
            0.0
        ) / static_cast<double>(samples.size());

    std::vector<double> sorted =
        samples;

    std::sort(
        sorted.begin(),
        sorted.end()
    );

    const std::size_t count =
        sorted.size();

    if (count % 2 == 0) {
        stats.median_ms =
            (
                sorted[count / 2 - 1] +
                sorted[count / 2]
            ) / 2.0;
    } else {
        stats.median_ms =
            sorted[count / 2];
    }

    const std::size_t p95_index =
        static_cast<std::size_t>(
            std::ceil(
                0.95 *
                static_cast<double>(count)
            )
        ) - 1;

    stats.p95_ms =
        sorted[
            std::min(
                p95_index,
                count - 1
            )
        ];

    stats.min_ms =
        sorted.front();

    stats.max_ms =
        sorted.back();

    double variance = 0.0;

    for (const double sample : samples) {
        const double difference =
            sample - stats.mean_ms;

        variance +=
            difference * difference;
    }

    variance /=
        static_cast<double>(
            samples.size()
        );

    stats.stddev_ms =
        std::sqrt(variance);

    return stats;
}

template <typename Callable>
BenchmarkStats benchmark_cuda(
    Callable&& operation,
    int warmup_iterations = 50,
    int measured_iterations = 500
) {
    if (
        warmup_iterations < 0 ||
        measured_iterations <= 0
    ) {
        throw std::invalid_argument(
            "Invalid benchmark iteration count"
        );
    }

    for (
        int i = 0;
        i < warmup_iterations;
        ++i
    ) {
        operation();
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    std::vector<double> samples;

    samples.reserve(
        measured_iterations
    );

    for (
        int i = 0;
        i < measured_iterations;
        ++i
    ) {
        CUDA_CHECK(
            cudaEventRecord(start)
        );

        operation();

        CUDA_CHECK(
            cudaEventRecord(stop)
        );

        CUDA_CHECK(
            cudaEventSynchronize(stop)
        );

        float elapsed_ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &elapsed_ms,
                start,
                stop
            )
        );

        samples.push_back(
            static_cast<double>(
                elapsed_ms
            )
        );
    }

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    return summarize_benchmark(
        samples,
        warmup_iterations
    );
}

} // namespace vortex
