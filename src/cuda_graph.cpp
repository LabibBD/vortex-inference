#include "vortex/cuda_graph.hpp"
#include "vortex/cuda_check.hpp"

#include <stdexcept>
#include <utility>

namespace vortex {

CudaGraph::~CudaGraph() {
    reset();
}

CudaGraph::CudaGraph(
    CudaGraph&& other
) noexcept
    : graph_(other.graph_),
      executable_(other.executable_) {

    other.graph_ = nullptr;
    other.executable_ = nullptr;
}

CudaGraph&
CudaGraph::operator=(
    CudaGraph&& other
) noexcept {
    if (this != &other) {
        reset();

        graph_ = other.graph_;
        executable_ =
            other.executable_;

        other.graph_ = nullptr;
        other.executable_ = nullptr;
    }

    return *this;
}

void CudaGraph::reset() noexcept {
    if (executable_ != nullptr) {
        cudaGraphExecDestroy(
            executable_
        );

        executable_ = nullptr;
    }

    if (graph_ != nullptr) {
        cudaGraphDestroy(
            graph_
        );

        graph_ = nullptr;
    }
}

void CudaGraph::capture(
    CudaStream& stream,
    const std::function<
        void(cudaStream_t)
    >& work
) {
    reset();

    CUDA_CHECK(
        cudaStreamBeginCapture(
            stream.get(),
            cudaStreamCaptureModeGlobal
        )
    );

    try {
        work(
            stream.get()
        );
    } catch (...) {
        cudaGraph_t discarded =
            nullptr;

        cudaStreamEndCapture(
            stream.get(),
            &discarded
        );

        if (discarded != nullptr) {
            cudaGraphDestroy(
                discarded
            );
        }

        throw;
    }

    CUDA_CHECK(
        cudaStreamEndCapture(
            stream.get(),
            &graph_
        )
    );

    if (graph_ == nullptr) {
        throw std::runtime_error(
            "CUDA graph capture produced a null graph"
        );
    }

    CUDA_CHECK(
        cudaGraphInstantiate(
            &executable_,
            graph_,
            nullptr,
            nullptr,
            0
        )
    );
}

void CudaGraph::launch(
    cudaStream_t stream
) const {
    if (executable_ == nullptr) {
        throw std::runtime_error(
            "Attempted to launch an uninitialized CUDA graph"
        );
    }

    CUDA_CHECK(
        cudaGraphLaunch(
            executable_,
            stream
        )
    );
}

bool CudaGraph::valid() const noexcept {
    return executable_ != nullptr;
}

} // namespace vortex
