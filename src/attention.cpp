#include "vortex/operators/attention.hpp"

#include "vortex/operators/softmax.hpp"
#include "vortex/operators/scaled_softmax.hpp"
#include "vortex/operators/transformer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

bool use_fused_scaled_softmax() noexcept {
    static const bool enabled = []() {
        const char* value =
            std::getenv(
                "VORTEX_ATTENTION_SOFTMAX_MODE"
            );

        if (value == nullptr) {
            return true;
        }

        return !(
            std::strcmp(
                value,
                "separate"
            ) == 0 ||
            std::strcmp(
                value,
                "legacy"
            ) == 0 ||
            std::strcmp(
                value,
                "0"
            ) == 0
        );
    }();

    return enabled;
}

void validate_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& output
) {
    if (
        query.device() != Device::CUDA ||
        key.device() != Device::CUDA ||
        value.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "Scaled attention requires CUDA tensors"
        );
    }

    if (
        query.dtype() != DataType::Float32 ||
        key.dtype() != DataType::Float32 ||
        value.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Scaled attention currently supports FP32 only"
        );
    }

    if (
        query.shape().size() != 3 ||
        key.shape().size() != 3 ||
        value.shape().size() != 3 ||
        output.shape().size() != 3
    ) {
        throw std::invalid_argument(
            "Scaled attention requires rank-3 tensors"
        );
    }

    const std::size_t batch =
        query.shape()[0];

    const std::size_t query_tokens =
        query.shape()[1];

    const std::size_t head_dim =
        query.shape()[2];

    const std::size_t key_tokens =
        key.shape()[1];

    const std::size_t value_dim =
        value.shape()[2];

    if (
        key.shape()[0] != batch ||
        value.shape()[0] != batch
    ) {
        throw std::invalid_argument(
            "Attention batch dimensions must match"
        );
    }

    if (key.shape()[2] != head_dim) {
        throw std::invalid_argument(
            "Query and key head dimensions must match"
        );
    }

    if (value.shape()[1] != key_tokens) {
        throw std::invalid_argument(
            "Key and value token counts must match"
        );
    }

    if (
        output.shape()[0] != batch ||
        output.shape()[1] != query_tokens ||
        output.shape()[2] != value_dim
    ) {
        throw std::invalid_argument(
            "Attention output shape is incorrect"
        );
    }

    if (head_dim == 0) {
        throw std::invalid_argument(
            "Attention head dimension cannot be zero"
        );
    }
}

} // namespace

void scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output
) {
    validate_attention(
        query,
        key,
        value,
        output
    );

    const std::size_t batch =
        query.shape()[0];

    const std::size_t query_tokens =
        query.shape()[1];

    const std::size_t head_dim =
        query.shape()[2];

    const std::size_t key_tokens =
        key.shape()[1];

    // K: [B, Tk, D]
    // K^T: [B, D, Tk]
    Tensor key_transposed(
        {
            batch,
            head_dim,
            key_tokens
        },
        DataType::Float32,
        Device::CUDA
    );

    // Scores: [B, Tq, Tk]
    Tensor scores(
        {
            batch,
            query_tokens,
            key_tokens
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor probabilities(
        {
            batch,
            query_tokens,
            key_tokens
        },
        DataType::Float32,
        Device::CUDA
    );

    transpose_last_two(
        key,
        key_transposed
    );

    batched_matmul(
        query,
        key_transposed,
        scores
    );

    const float scale_factor =
        1.0f /
        std::sqrt(
            static_cast<float>(
                head_dim
            )
        );

    if (
        use_fused_scaled_softmax()
    ) {
        scaled_softmax(
            scores,
            scale_factor,
            probabilities
        );
    } else {
        scale(
            scores,
            scale_factor,
            scores
        );

        softmax(
            scores,
            probabilities
        );
    }

    batched_matmul(
        probabilities,
        value,
        output
    );
}

} // namespace vortex::ops
