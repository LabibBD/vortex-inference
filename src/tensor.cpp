#include "vortex/tensor.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <new>
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

    allocate();
}

Tensor::~Tensor() {
    release();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_),
      numel_(other.numel_),
      bytes_(other.bytes_),
      data_(other.data_) {

    // The moved-from Tensor no longer owns the memory.
    other.numel_ = 0;
    other.bytes_ = 0;
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Release anything this Tensor already owns.
        release();

        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        numel_ = other.numel_;
        bytes_ = other.bytes_;
        data_ = other.data_;

        // Transfer ownership away from other.
        other.numel_ = 0;
        other.bytes_ = 0;
        other.data_ = nullptr;
    }

    return *this;
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
    return data_;
}

const void* Tensor::data() const noexcept {
    return data_;
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
        CUDA_CHECK(cudaMemcpy(
            data_,
            source,
            bytes,
            cudaMemcpyHostToDevice
        ));
    } else {
        std::memcpy(
            data_,
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
        CUDA_CHECK(cudaMemcpy(
            destination,
            data_,
            bytes,
            cudaMemcpyDeviceToHost
        ));
    } else {
        std::memcpy(
            destination,
            data_,
            bytes
        );
    }
}

void Tensor::allocate() {
    if (bytes_ == 0) {
        return;
    }

    if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMalloc(
            &data_,
            bytes_
        ));
    } else {
        data_ = std::malloc(bytes_);

        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
    }
}

void Tensor::release() noexcept {
    if (data_ == nullptr) {
        return;
    }

    if (device_ == Device::CUDA) {
        cudaFree(data_);
    } else {
        std::free(data_);
    }

    data_ = nullptr;
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
