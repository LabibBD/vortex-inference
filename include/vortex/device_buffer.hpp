#pragma once

#include <cstddef>

namespace vortex {

class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(
        std::size_t bytes
    );

    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void allocate(std::size_t bytes);
    void release() noexcept;

    void copy_from_host(
        const void* source,
        std::size_t bytes
    );

    void copy_to_host(
        void* destination,
        std::size_t bytes
    ) const;

    void* data() noexcept;
    const void* data() const noexcept;

    std::size_t size() const noexcept;
    bool empty() const noexcept;

private:
    void* data_{nullptr};
    std::size_t size_{0};
};

} // namespace vortex
