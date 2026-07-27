#pragma once

#include "vortex/device_buffer.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace vortex {

enum class DataType {
    Float32,
    Float16
};

enum class Device {
    CPU,
    CUDA
};

class Tensor {
public:
    Tensor(
        std::vector<std::size_t> shape,
        DataType dtype,
        Device device
    );

    ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    const std::vector<std::size_t>& shape() const noexcept;

    DataType dtype() const noexcept;
    Device device() const noexcept;

    std::size_t numel() const noexcept;
    std::size_t bytes() const noexcept;

    void* data() noexcept;
    const void* data() const noexcept;

    void copy_from_host(
        const void* source,
        std::size_t bytes
    );

    void copy_to_host(
        void* destination,
        std::size_t bytes
    ) const;

private:
    static std::size_t element_size(DataType dtype);

    std::vector<std::size_t> shape_;

    DataType dtype_;
    Device device_;

    std::size_t numel_{0};
    std::size_t bytes_{0};

    std::unique_ptr<std::byte[]> cpu_data_;
    DeviceBuffer device_buffer_;
};

} // namespace vortex
