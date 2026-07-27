#include "vortex/device_buffer.hpp"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N = 4096;
    constexpr std::size_t BYTES = N * sizeof(float);

    std::cout << "=================================\n";
    std::cout << " VORTEX DeviceBuffer Test\n";
    std::cout << "=================================\n\n";

    std::vector<float> input(N);
    std::vector<float> output(N);

    for (std::size_t i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i) * 0.25f;
    }

    // --------------------------------------------------
    // Allocation
    // --------------------------------------------------

    DeviceBuffer buffer(BYTES);

    const bool allocation_ok =
        !buffer.empty() &&
        buffer.data() != nullptr &&
        buffer.size() == BYTES;

    std::cout << "Allocation:       "
              << (allocation_ok ? "PASSED" : "FAILED")
              << '\n';

    std::cout << "Buffer size:      "
              << buffer.size()
              << " bytes\n";

    // --------------------------------------------------
    // Host -> GPU -> Host transfer
    // --------------------------------------------------

    buffer.copy_from_host(
        input.data(),
        BYTES
    );

    buffer.copy_to_host(
        output.data(),
        BYTES
    );

    float max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        const float error =
            std::fabs(input[i] - output[i]);

        if (error > max_error) {
            max_error = error;
        }
    }

    const bool transfer_ok =
        max_error == 0.0f;

    std::cout << "Transfer:         "
              << (transfer_ok ? "PASSED" : "FAILED")
              << '\n';

    std::cout << "Maximum error:    "
              << max_error
              << '\n';

    // --------------------------------------------------
    // Move constructor
    // --------------------------------------------------

    void* original_pointer =
        buffer.data();

    DeviceBuffer moved_buffer =
        std::move(buffer);

    const bool move_ok =
        buffer.empty() &&
        buffer.data() == nullptr &&
        buffer.size() == 0 &&
        moved_buffer.data() == original_pointer &&
        moved_buffer.size() == BYTES;

    std::cout << "Move ownership:   "
              << (move_ok ? "PASSED" : "FAILED")
              << '\n';

    // --------------------------------------------------
    // Reallocation
    // --------------------------------------------------

    constexpr std::size_t NEW_N = 8192;
    constexpr std::size_t NEW_BYTES =
        NEW_N * sizeof(float);

    moved_buffer.allocate(NEW_BYTES);

    const bool reallocation_ok =
        moved_buffer.data() != nullptr &&
        moved_buffer.size() == NEW_BYTES;

    std::cout << "Reallocation:     "
              << (reallocation_ok ? "PASSED" : "FAILED")
              << '\n';

    std::cout << "New buffer size:  "
              << moved_buffer.size()
              << " bytes\n";

    // --------------------------------------------------
    // Manual release
    // --------------------------------------------------

    moved_buffer.release();

    const bool release_ok =
        moved_buffer.empty() &&
        moved_buffer.data() == nullptr &&
        moved_buffer.size() == 0;

    std::cout << "Release:          "
              << (release_ok ? "PASSED" : "FAILED")
              << '\n';

    const bool passed =
        allocation_ok &&
        transfer_ok &&
        move_ok &&
        reallocation_ok &&
        release_ok;

    if (passed) {
        std::cout
            << "\n[VORTEX] DeviceBuffer test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] DeviceBuffer test FAILED\n";

    return 1;
}
