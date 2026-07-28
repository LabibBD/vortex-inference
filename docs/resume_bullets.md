# VORTEX Résumé Bullets

- Built **VORTEX**, a C++17/CUDA neural-network inference runtime with static graph execution, GPU memory pooling, tensor lifetime analysis, ONNX conversion, FP32/FP16 operators, and transformer self-attention.

- Profiled and optimized CUDA execution using Nsight, reducing attention from 16 to 15 kernels and achieving **0.034802 ms median latency** on a fixed transformer workload—**3.065× faster than PyTorch eager** and **2.439× faster than compiled PyTorch** on an RTX 4060 Laptop GPU.

- Added randomized PyTorch numerical validation, experimental dynamic INT8 inference, a JSON C++ CLI, and a Python bridge for HELIOS outcome verification.
