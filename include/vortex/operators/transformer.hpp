#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

/*
 * Supports:
 *
 * [rows, columns] -> [columns, rows]
 *
 * and:
 *
 * [batch, rows, columns]
 *     ->
 * [batch, columns, rows]
 */
void transpose_last_two(
    const Tensor& input,
    Tensor& output
);

void scale(
    const Tensor& input,
    float factor,
    Tensor& output
);

/*
 * A: [batch, M, K]
 * B: [batch, K, N]
 * C: [batch, M, N]
 */
void batched_matmul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

} // namespace vortex::ops
