# VORTEX Phase 25 — Numerical Validation

## Coverage

VORTEX was validated for:

- Numerically stable Softmax with extreme positive and negative inputs
- Fused scaled-Softmax
- Rank-2 and rank-3 LayerNorm
- Scaled dot-product attention with irregular dimensions
- FP16 ReLU and GELU
- Explicit NaN and infinity detection
- Repeated-execution determinism
- Complete transformer inference against PyTorch references

## Randomized Transformer Validation

The complete transformer block was tested with eight independent seeds and
input distributions:

1. Small normal values
2. Standard normal values
3. Large normal values
4. Standard uniform values
5. Large uniform values
6. Near-constant values
7. Sparse values
8. Alternating large positive and negative values

Each case used independently initialized parameters and a separately generated
PyTorch FP32 reference.

Each configuration was executed 21 times:

- One numerical-comparison execution
- Twenty repeated determinism executions

## Acceptance Thresholds

- Full-transformer maximum absolute error: 5e-3
- Full-transformer worst mean absolute error: 5e-4
- Repeated-execution maximum error: 1e-7
- Attention maximum absolute error: 8e-4
- Attention mean absolute error: 8e-5
- FP16 ReLU maximum error: 2e-3
- FP16 GELU maximum error: 1e-2
- Softmax row-sum error: 2e-5

## Result

All 28 CTest tests passed.

All randomized transformer cases:

- Produced finite outputs
- Matched their PyTorch references within tolerance
- Produced deterministic repeated results
- Passed under optimized boundary synchronization
- Passed with fused attention scaled-Softmax enabled

Measured values are preserved in:

- `results/phase25_numerical_validation.txt`
- `results/phase25_randomized_transformer.txt`
