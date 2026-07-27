#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int device_count = 0;

    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "[VORTEX] No CUDA GPU detected\n";
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "===============================\n";
    std::cout << " VORTEX CUDA Environment Check\n";
    std::cout << "===============================\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "VRAM: "
              << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
              << " GB\n\n";

    const int n = 1024;
    const size_t bytes = n * sizeof(float);

    float h_a[n];
    float h_b[n];
    float h_c[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaError_t kernel_error = cudaGetLastError();

    if (kernel_error != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(kernel_error) << "\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;

    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];

        if (h_c[i] != expected) {
            correct = false;
            break;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Vector size: " << n << "\n";
    std::cout << "Threads/block: " << threads << "\n";
    std::cout << "Blocks: " << blocks << "\n";

    if (correct) {
        std::cout << "\n[VORTEX] CUDA kernel PASSED\n";
        return 0;
    }

    std::cout << "\n[VORTEX] CUDA kernel FAILED\n";
    return 1;
}
