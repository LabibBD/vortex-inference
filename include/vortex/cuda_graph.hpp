#pragma once

#include "vortex/cuda_stream.hpp"

#include <cuda_runtime.h>

#include <functional>

namespace vortex {

class CudaGraph {
public:
    CudaGraph() = default;
    ~CudaGraph();

    CudaGraph(
        const CudaGraph&
    ) = delete;

    CudaGraph&
    operator=(
        const CudaGraph&
    ) = delete;

    CudaGraph(
        CudaGraph&& other
    ) noexcept;

    CudaGraph&
    operator=(
        CudaGraph&& other
    ) noexcept;

    void capture(
        CudaStream& stream,
        const std::function<
            void(cudaStream_t)
        >& work
    );

    void launch(
        cudaStream_t stream
    ) const;

    bool valid() const noexcept;

private:
    void reset() noexcept;

    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t executable_{nullptr};
};

} // namespace vortex
