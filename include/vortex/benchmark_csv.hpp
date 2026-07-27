#pragma once

#include "vortex/benchmark.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace vortex {

inline void append_benchmark_csv(
    const std::string& path,
    const std::string& operator_name,
    std::size_t elements,
    const BenchmarkStats& stats,
    double throughput_m_elements_s
) {
    const std::filesystem::path file_path(path);

    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(
            file_path.parent_path()
        );
    }

    const bool exists =
        std::filesystem::exists(file_path);

    cudaDeviceProp properties{};
    CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            0
        )
    );

    std::ofstream file(
        path,
        std::ios::app
    );

    if (!file) {
        throw std::runtime_error(
            "Unable to open benchmark CSV"
        );
    }

    if (!exists) {
        file
            << "operator,"
            << "elements,"
            << "dtype,"
            << "gpu,"
            << "warmup,"
            << "iterations,"
            << "mean_ms,"
            << "median_ms,"
            << "p95_ms,"
            << "stddev_ms,"
            << "min_ms,"
            << "max_ms,"
            << "throughput_m_elements_s\n";
    }

    file
        << operator_name << ','
        << elements << ','
        << "FP32,"
        << '"' << properties.name << '"' << ','
        << stats.warmup_iterations << ','
        << stats.measured_iterations << ','
        << std::setprecision(10)
        << stats.mean_ms << ','
        << stats.median_ms << ','
        << stats.p95_ms << ','
        << stats.stddev_ms << ','
        << stats.min_ms << ','
        << stats.max_ms << ','
        << throughput_m_elements_s
        << '\n';
}

} // namespace vortex
