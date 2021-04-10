#include "cpu_gpu_image.hpp"

#include "cuda.h"

namespace {

void check(cudaError_t const cuda_status, std::string const& stage)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << " in " << stage << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

}


CpuGpuImage::CpuGpuImage(int32_t const width, int32_t const height, std::string const& name)
: width_(width),
  height_(height),
  cpu_mat_(width, height, CV_8UC3),
  name_(name)
{
cudaError_t cuda_status = cudaSuccess;

cuda_status = cudaMalloc((void**) &gpu_ptr_, width * height * 3);
check(cuda_status, "GPU buffer allocation " + name_);
}

CpuGpuImage::~CpuGpuImage()
{
    cuda_status = cudaFree((void*) gpu_ptr_);
    check(cuda_status, "GPU buffer release" + name_);
}

void CpuGpuImage::copyToGpu()
{
    cudaError_t cuda_status = cudaMemcpy((void*) gpu_ptr_, (void const *) cpu_mat_.ptr(), width_ * height_ * 3, cudaMemcpyHostToDevice);
    check(cuda_status, "CPU --> GPU copy" + name_);
}

void CpuGpuImage::copyFromGpu()
{
    cudaError_t cuda_status = cudaMemcpy((void*) cpu_mat_.ptr(), (void const *) gpu_ptr_, width_ * height_ * 3, cudaMemcpyDeviceToHost);
    check(cuda_status, "GPU --> CPU copy" + name_);
}
