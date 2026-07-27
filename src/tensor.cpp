#include "vortex/tensor.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace vortex {

Tensor::Tensor(
    std::vector<std::size_t> shape,
    DataType dtype,
    Device device
)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device) {

    if (shape_.empty()) {
        throw std::invalid_argument(
            "Tensor shape cannot be empty"
        );
    }

    numel_ = 1;

    for (const std::size_t dimension : shape_) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be greater than zero"
            );
        }

        numel_ *= dimension;
    }

    bytes_ = numel_ * element_size(dtype_);

    if (device_ == Device::CUDA) {
        device_buffer_.allocate(bytes_);
    } else {
        cpu_data_ =
            std::make_unique<std::byte[]>(bytes_);
    }
}

const std::vector<std::size_t>&
Tensor::shape() const noexcept {
    return shape_;
}

DataType Tensor::dtype() const noexcept {
    return dtype_;
}

Device Tensor::device() const noexcept {
    return device_;
}

std::size_t Tensor::numel() const noexcept {
    return numel_;
}

std::size_t Tensor::bytes() const noexcept {
    return bytes_;
}

void* Tensor::data() noexcept {
    if (device_ == Device::CUDA) {
        return device_buffer_.data();
    }

    return cpu_data_.get();
}

const void* Tensor::data() const noexcept {
    if (device_ == Device::CUDA) {
        return device_buffer_.data();
    }

    return cpu_data_.get();
}

void Tensor::copy_from_host(
    const void* source,
    std::size_t bytes
) {
    if (source == nullptr) {
        throw std::invalid_argument(
            "Source pointer cannot be null"
        );
    }

    if (bytes > bytes_) {
        throw std::out_of_range(
            "Host-to-tensor copy exceeds tensor size"
        );
    }

    if (device_ == Device::CUDA) {
        device_buffer_.copy_from_host(
            source,
            bytes
        );
    } else {
        std::memcpy(
            cpu_data_.get(),
            source,
            bytes
        );
    }
}

void Tensor::copy_to_host(
    void* destination,
    std::size_t bytes
) const {
    if (destination == nullptr) {
        throw std::invalid_argument(
            "Destination pointer cannot be null"
        );
    }

    if (bytes > bytes_) {
        throw std::out_of_range(
            "Tensor-to-host copy exceeds tensor size"
        );
    }

    if (device_ == Device::CUDA) {
        device_buffer_.copy_to_host(
            destination,
            bytes
        );
    } else {
        std::memcpy(
            destination,
            cpu_data_.get(),
            bytes
        );
    }
}

std::size_t Tensor::element_size(
    DataType dtype
) {
    switch (dtype) {
        case DataType::Float32:
            return sizeof(float);
    }

    throw std::runtime_error(
        "Unsupported VORTEX data type"
    );
}

} // namespace vortex
