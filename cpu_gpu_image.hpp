#ifndef MANOJ_CPU_GPU_IMAGE_CUH
#define MANOJ_CPU_GPU_IMAGE_CUH

#include <string>

#include "opencv2/core.hpp"

class CpuGpuImage
{
  public:
    CpuGpuImage(int32_t width, int32_t height, std::string const& name);
    ~CpuGpuImage();

    int32_t width() const;
    int32_t height() const;
    uchar3* gpuPtr() const;
    cv::Mat& cpuMat();
    cv::Mat const& cpuMat() const;

    void copyToGpu();
    void copyFromGpu();
    
  private:
    cv::Mat cpu_mat_;
    uchar3 *gpu_ptr_;
    int32_t width_;
    int32_t height_;
    std::string name_;
};

// --- Inline defs ---

inline int32_t CpuGpuImage::width() const {
    return width_;
}

inline int32_t CpuGpuImage::height() const {
    return height_;
}

inline uchar3* CpuGpuImage::gpuPtr() const {
    return gpu_ptr_;
}

inline cv::Mat& CpuGpuImage::cpuMat() {
    return cpu_mat_;
}

inline cv::Mat const& CpuGpuImage::cpuMat() const {
    return cpu_mat_;
}

#endif // MANOJ_CPU_GPU_IMAGE_CUH
