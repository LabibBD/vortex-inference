#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

/*
 * Scaled dot-product attention.
 *
 * query:  [batch, query_tokens, head_dim]
 * key:    [batch, key_tokens, head_dim]
 * value:  [batch, key_tokens, value_dim]
 * output: [batch, query_tokens, value_dim]
 */
void scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output
);

} // namespace vortex::ops
