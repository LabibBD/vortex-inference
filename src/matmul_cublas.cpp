#include "vortex/operators/matmul.hpp"

#include "vortex/cublas_check.hpp"
#include "vortex/cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace vortex::ops {

namespace {

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(
            cublasCreate(&handle_)
        );

        CUBLAS_CHECK(
            cublasSetMathMode(
                handle_,
                CUBLAS_DEFAULT_MATH
            )
        );
    }

    ~CublasHandle() {
        cublasDestroy(handle_);
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_{};
};

cublasHandle_t get_handle() {
    static CublasHandle handle;
    return handle.get();
}

void validate_matmul(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output
) {
    if (
        a.device() != Device::CUDA ||
        b.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "cuBLAS MatMul requires CUDA tensors"
        );
    }

    if (
        a.dtype() != DataType::Float32 ||
        b.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "cuBLAS MatMul currently supports FP32 only"
        );
    }

    if (
        a.shape().size() != 2 ||
        b.shape().size() != 2 ||
        output.shape().size() != 2
    ) {
        throw std::invalid_argument(
            "cuBLAS MatMul requires 2D tensors"
        );
    }

    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(
            "MatMul inner dimensions do not match"
        );
    }

    if (
        output.shape()[0] != a.shape()[0] ||
        output.shape()[1] != b.shape()[1]
    ) {
        throw std::invalid_argument(
            "MatMul output shape is incorrect"
        );
    }
}

} // namespace

void matmul_cublas(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    validate_matmul(
        a,
        b,
        output
    );

    const int m =
        static_cast<int>(a.shape()[0]);

    const int k =
        static_cast<int>(a.shape()[1]);

    const int n =
        static_cast<int>(b.shape()[1]);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    /*
     * VORTEX stores matrices in row-major layout.
     *
     * cuBLAS uses column-major semantics, so:
     *
     * C = A * B
     *
     * is evaluated as:
     *
     * C^T = B^T * A^T
     *
     * by swapping A/B and M/N in the cuBLAS call.
     */

    CUBLAS_CHECK(
        cublasSgemm(
            get_handle(),

            CUBLAS_OP_N,
            CUBLAS_OP_N,

            n,
            m,
            k,

            &alpha,

            static_cast<const float*>(
                b.data()
            ),
            n,

            static_cast<const float*>(
                a.data()
            ),
            k,

            &beta,

            static_cast<float*>(
                output.data()
            ),
            n
        )
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}

} // namespace vortex::ops
