#include "vortex/cuda_stream.hpp"
#include "vortex/cuda_check.hpp"

#include <utility>

namespace vortex {

CudaStream::CudaStream() {
    CUDA_CHECK(
        cudaStreamCreateWithFlags(
            &stream_,
            cudaStreamNonBlocking
        )
    );
}

CudaStream::~CudaStream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(
            stream_
        );
    }
}

CudaStream::CudaStream(
    CudaStream&& other
) noexcept
    : stream_(other.stream_) {

    other.stream_ =
        nullptr;
}

CudaStream&
CudaStream::operator=(
    CudaStream&& other
) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cudaStreamDestroy(
                stream_
            );
        }

        stream_ =
            other.stream_;

        other.stream_ =
            nullptr;
    }

    return *this;
}

cudaStream_t
CudaStream::get() const noexcept {
    return stream_;
}

void CudaStream::synchronize() const {
    CUDA_CHECK(
        cudaStreamSynchronize(
            stream_
        )
    );
}

} // namespace vortex
