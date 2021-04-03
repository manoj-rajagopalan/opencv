#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include <cuda.h>

using std::cout;
using std::endl;

int constexpr kN = 1000;
std::mt19937_64 rand_engine;

void cudaCheckSuccess(cudaError_t const cuda_status, std::string const& message)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << ": " << message << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

__constant__ float filter_gpu[9];

__global__ void modulate(float *data, int N)
{
    int const tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N) {
        data[tid] *= filter_gpu[tid % 9];
    }
}

int main(void)
{
    std::vector<float> v(kN);
    std::uniform_real_distribution<float> rand_gen(0.0, 1.0);
    std::generate(v.begin(), v.end(), [&](){ return rand_gen(rand_engine); });

    float *gpu_v;
    cudaError_t cuda_status = cudaMalloc((void**) &gpu_v, kN * sizeof(float));
    cudaCheckSuccess(cuda_status, "Unable to cudaMalloc");

    cuda_status = cudaMemcpy((void*) gpu_v, (const void*) v.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckSuccess(cuda_status, "Unable to memcpy into GPU");

    std::vector<float> filter(9);
    std::iota(filter.begin(), filter.end(), 1.0);
    cuda_status = cudaMemcpyToSymbol(filter_gpu, (const void*) filter.data(), 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaCheckSuccess(cuda_status, "Unable to cudaMemcpyToSymbol");

    modulate<<<(v.size() + 99 ) / 100, 100>>>(gpu_v, kN);
    cuda_status = cudaGetLastError();
    cudaCheckSuccess(cuda_status, "Unable to launch kernel");

    std::vector<float> v2(v.size(), 0.0f);
    cuda_status = cudaMemcpy((void*) v2.data(), (const void*) gpu_v, kN * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckSuccess(cuda_status, "Unable to copy data out of GPU");

    std::vector<float> v3(v.size(), 0.0f);
    for(int i = 0;  i < v.size(); ++i) {
        v3[i] = filter[i % filter.size()] * v[i];
    }

    assert(v2 == v3 && "FAIL");

    return 0;
}