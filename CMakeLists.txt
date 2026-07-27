cmake_minimum_required(VERSION 3.24)

project(
    VORTEX
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 Laptop GPU - Ada Lovelace, Compute Capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(
    vortex_cuda_check
    examples/cuda_check.cu
)

target_compile_options(
    vortex_cuda_check
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(
    vortex_vector_add
    examples/vector_add.cu
)

target_include_directories(
    vortex_vector_add
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    vortex_vector_add
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(
    vortex_scalar_mul
    examples/scalar_mul.cu
)

target_include_directories(
    vortex_scalar_mul
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    vortex_scalar_mul
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(
    vortex_relu
    examples/relu.cu
)

target_include_directories(
    vortex_relu
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    vortex_relu
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# ============================================================
# VORTEX Core Runtime
# ============================================================

find_package(CUDAToolkit REQUIRED)

add_library(
    vortex_core
    src/tensor.cpp
    src/device_buffer.cpp
    src/matmul_cublas.cpp
    kernels/elementwise.cu
    kernels/activations.cu
    kernels/reductions.cu
    kernels/softmax.cu
    kernels/matmul_naive.cu
    kernels/matmul_tiled.cu
    kernels/linear.cu
    kernels/layernorm.cu
)

target_include_directories(
    vortex_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    vortex_core
    PUBLIC
    CUDA::cudart
)

add_executable(
    vortex_tensor_test
    tests/tensor_test.cpp
)

target_link_libraries(
    vortex_tensor_test
    PRIVATE
    vortex_core
)

add_executable(
    vortex_device_buffer_test
    tests/device_buffer_test.cpp
)

target_link_libraries(
    vortex_device_buffer_test
    PRIVATE
    vortex_core
)

add_executable(
    vortex_elementwise_test
    tests/elementwise_test.cpp
)

target_link_libraries(
    vortex_elementwise_test
    PRIVATE
    vortex_core
)

add_executable(
    vortex_activation_test
    tests/activation_test.cpp
)

target_link_libraries(
    vortex_activation_test
    PRIVATE
    vortex_core
)

add_executable(
    vortex_reduction_test
    tests/reduction_test.cpp
)

target_link_libraries(
    vortex_reduction_test
    PRIVATE
    vortex_core
)

add_executable(
    vortex_softmax_test
    tests/softmax_test.cpp
)

target_link_libraries(
    vortex_softmax_test
    PRIVATE
    vortex_core
)

# ============================================================
# Automated Test Suite
# ============================================================

enable_testing()

add_test(
    NAME tensor
    COMMAND vortex_tensor_test
)

add_test(
    NAME device_buffer
    COMMAND vortex_device_buffer_test
)

add_test(
    NAME elementwise
    COMMAND vortex_elementwise_test
)

add_test(
    NAME activations
    COMMAND vortex_activation_test
)

add_test(
    NAME reductions
    COMMAND vortex_reduction_test
)

add_test(
    NAME softmax
    COMMAND vortex_softmax_test
)

# ============================================================
# Benchmarks
# ============================================================

add_executable(
    vortex_activation_benchmark
    benchmarks/activation_benchmark.cpp
)

target_link_libraries(
    vortex_activation_benchmark
    PRIVATE
    vortex_core
)

add_executable(
    vortex_matmul_test
    tests/matmul_test.cpp
)

target_link_libraries(
    vortex_matmul_test
    PRIVATE
    vortex_core
)

add_test(
    NAME matmul
    COMMAND vortex_matmul_test
)

add_executable(
    vortex_matmul_benchmark
    benchmarks/matmul_benchmark.cpp
)

target_link_libraries(
    vortex_matmul_benchmark
    PRIVATE
    vortex_core
)

add_executable(
    vortex_matmul_profile
    benchmarks/matmul_profile.cpp
)

target_link_libraries(
    vortex_matmul_profile
    PRIVATE
    vortex_core
)

target_link_libraries(
    vortex_core
    PUBLIC
    CUDA::cublas
)

add_executable(
    vortex_linear_test
    tests/linear_test.cpp
)

target_link_libraries(
    vortex_linear_test
    PRIVATE
    vortex_core
)

add_test(
    NAME linear
    COMMAND vortex_linear_test
)

add_executable(
    vortex_layernorm_test
    tests/layernorm_test.cpp
)

target_link_libraries(
    vortex_layernorm_test
    PRIVATE
    vortex_core
)

add_test(
    NAME layernorm
    COMMAND vortex_layernorm_test
)


