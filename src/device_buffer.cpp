#include "vortex/device_buffer.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace vortex {

DeviceBuffer::DeviceBuffer(
    std::size_t bytes
) {
    allocate(bytes);
}

DeviceBuffer::~DeviceBuffer() {
    release();
}

DeviceBuffer::DeviceBuffer(
    DeviceBuffer&& other
) noexcept
    : data_(other.data_),
      size_(other.size_) {

    other.data_ = nullptr;
    other.size_ = 0;
}

DeviceBuffer&
DeviceBuffer::operator=(
    DeviceBuffer&& other
) noexcept {

    if (this != &other) {
        release();

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;
    }

    return *this;
}

void DeviceBuffer::allocate(
    std::size_t bytes
) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "DeviceBuffer allocation size must be greater than zero"
        );
    }

    // Avoid silently leaking an existing allocation.
    release();

    CUDA_CHECK(
        cudaMalloc(
            &data_,
            bytes
        )
    );

    size_ = bytes;
}

void DeviceBuffer::release() noexcept {
    if (data_ == nullptr) {
        size_ = 0;
        return;
    }

    cudaFree(data_);

    data_ = nullptr;
    size_ = 0;
}

void DeviceBuffer::copy_from_host(
    const void* source,
    std::size_t bytes
) {
    if (source == nullptr) {
        throw std::invalid_argument(
            "Source pointer cannot be null"
        );
    }

    if (bytes > size_) {
        throw std::out_of_range(
            "Host-to-device copy exceeds buffer size"
        );
    }

    CUDA_CHECK(
        cudaMemcpy(
            data_,
            source,
            bytes,
            cudaMemcpyHostToDevice
        )
    );
}

void DeviceBuffer::copy_to_host(
    void* destination,
    std::size_t bytes
) const {
    if (destination == nullptr) {
        throw std::invalid_argument(
            "Destination pointer cannot be null"
        );
    }

    if (bytes > size_) {
        throw std::out_of_range(
            "Device-to-host copy exceeds buffer size"
        );
    }

    CUDA_CHECK(
        cudaMemcpy(
            destination,
            data_,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );
}

void* DeviceBuffer::data() noexcept {
    return data_;
}

const void*
DeviceBuffer::data() const noexcept {
    return data_;
}

std::size_t
DeviceBuffer::size() const noexcept {
    return size_;
}

bool DeviceBuffer::empty() const noexcept {
    return data_ == nullptr;
}

} // namespace vortex
