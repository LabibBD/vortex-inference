#pragma once

#include <cstddef>
#include <vector>

namespace vortex {

enum class DataType {
    Float32
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

    ~Tensor();

    // Do not allow copying of owned memory.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Allow ownership to be transferred safely.
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

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
    void allocate();
    void release() noexcept;

    static std::size_t element_size(DataType dtype);

    std::vector<std::size_t> shape_;

    DataType dtype_;
    Device device_;

    std::size_t numel_{0};
    std::size_t bytes_{0};

    void* data_{nullptr};
};

} // namespace vortex
