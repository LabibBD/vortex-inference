#include "vortex/tensor.hpp"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N = 1024;

    std::cout << "===============================\n";
    std::cout << " VORTEX Tensor Test\n";
    std::cout << "===============================\n\n";

    // --------------------------------------------------
    // CUDA Tensor
    // --------------------------------------------------

    std::vector<float> input(N);

    for (std::size_t i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i) * 0.5f;
    }

    Tensor gpu_tensor(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    std::cout << "CUDA tensor elements: "
              << gpu_tensor.numel()
              << '\n';

    std::cout << "CUDA tensor bytes:    "
              << gpu_tensor.bytes()
              << '\n';

    gpu_tensor.copy_from_host(
        input.data(),
        input.size() * sizeof(float)
    );

    std::vector<float> output(N);

    gpu_tensor.copy_to_host(
        output.data(),
        output.size() * sizeof(float)
    );

    float max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        const float error =
            std::fabs(input[i] - output[i]);

        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "CUDA transfer error:  "
              << max_error
              << '\n';

    // --------------------------------------------------
    // CPU Tensor
    // --------------------------------------------------

    Tensor cpu_tensor(
        {2, 3, 4},
        DataType::Float32,
        Device::CPU
    );

    std::cout << "\nCPU tensor shape:      2 x 3 x 4\n";

    std::cout << "CPU tensor elements:   "
              << cpu_tensor.numel()
              << '\n';

    std::cout << "CPU tensor bytes:      "
              << cpu_tensor.bytes()
              << '\n';

    // Test CPU copy as well.
    std::vector<float> cpu_input(24, 3.14f);
    std::vector<float> cpu_output(24);

    cpu_tensor.copy_from_host(
        cpu_input.data(),
        cpu_input.size() * sizeof(float)
    );

    cpu_tensor.copy_to_host(
        cpu_output.data(),
        cpu_output.size() * sizeof(float)
    );

    bool cpu_copy_correct = true;

    for (std::size_t i = 0; i < cpu_input.size(); ++i) {
        if (cpu_input[i] != cpu_output[i]) {
            cpu_copy_correct = false;
            break;
        }
    }

    std::cout << "CPU copy:              "
              << (cpu_copy_correct ? "PASSED" : "FAILED")
              << '\n';

    // --------------------------------------------------
    // Move semantics
    // --------------------------------------------------

    void* original_pointer = gpu_tensor.data();

    Tensor moved_tensor =
        std::move(gpu_tensor);

    const bool move_correct =
        moved_tensor.data() == original_pointer &&
        gpu_tensor.data() == nullptr &&
        moved_tensor.numel() == N;

    std::cout << "\nMove ownership:         "
              << (move_correct ? "PASSED" : "FAILED")
              << '\n';

    // --------------------------------------------------
    // Final validation
    // --------------------------------------------------

    const bool passed =
        max_error == 0.0f &&
        cpu_tensor.numel() == 24 &&
        cpu_tensor.bytes() == 96 &&
        cpu_copy_correct &&
        move_correct;

    if (passed) {
        std::cout
            << "\n[VORTEX] Tensor test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Tensor test FAILED\n";

    return 1;
}
