# VORTEX Architecture

## Model Loading

The ONNX converter produces a native package containing `model.vtx` and `weights.bin`. `ModelLoader` reconstructs the graph and loads constants into the executor.

## Memory

CUDA tensors use reusable memory-pool blocks. Lifetime analysis identifies the last consumer of every intermediate tensor so its memory can be reused.

## Execution

`GraphExecutor` dispatches operators in graph order, releases dead tensors, and synchronizes at the inference boundary. Per-kernel synchronization remains available for debugging.

## Transformer

The transformer path contains Q/K/V projections, scaled dot-product attention, output projection, residual connections, LayerNorm, and a two-layer GELU feed-forward network. The optimized attention implementation fuses scaling and Softmax into one CUDA kernel.

## HELIOS

The Python backend invokes the C++ inference CLI with a 256-element FP32 feature vector. The output contains probabilities for `SUCCESS`, `FAILURE`, and `UNCERTAIN`.
