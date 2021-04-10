#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "opencv2/imgcodecs.hpp"

#include "cuda_histogram.cuh"

namespace {

void check(cudaError_t const cuda_status, std::string const& stage)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << " in " << stage << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

struct CudaThreadConfig
{
    explicit CudaThreadConfig(cv::Size2i const& bounding_box_extents)
    : bb_size(bounding_box_extents)
    {
        int const w = bb_size.width;
        int const h = bb_size.height;
        int const block_width = 32;
        int const block_height = 32;
        int const grid_width = (w + 31) / 32;
        int const grid_height = (h + 31) / 32;

        block.x = block_width;
        block.y = block_height;
        block.z = 1;

        grid.x = grid_width;
        grid.y = grid_height;
        grid.z = 1;
    }

    float fragmentation() const {
        int const bb_area = bb_size.width * bb_size.height;
        int const grid_area = (grid.x * block.x) * (grid.y * block.y);
        float const result = 1.0f - float(bb_area) / float(grid_area);
        return result;
    }

    cv::Size2i bb_size;
    dim3 grid;
    dim3 block;
};

} // namespace

void rgbHistogramCpu(cv::Mat const& img, int32_t r_hist[256], int32_t g_hist[256], int32_t b_hist[256])
{
    for(int n = 0; n < 256; ++n) {
        r_hist[n] = 0;
        g_hist[n] = 0;
        b_hist[n] = 0;
    }
    for(int j = 0; j < img.rows; ++j) {
        for(int i = 0; i < img.cols; ++i) {
            uchar3 const pixel = *( ((uchar3*) img.data) + j * img.cols + i );
            ++r_hist[pixel.x];
            ++g_hist[pixel.y];
            ++b_hist[pixel.z];
        }
    }
}

void compareHistograms(int32_t calculated_hist[256], int32_t gold_hist[256], std::string const& name)
{
    std::ofstream outfile(name + "_hist.csv");
    int num_mismatches = 0;
    for(int i = 0; i < 256; ++i) {
        outfile << std::setw(5) << i << std::setw(8) << gold_hist[i] << std::setw(8) << calculated_hist[i] << '\n';
        num_mismatches = (calculated_hist[i] != gold_hist[i]) ? 1 : 0;
    }
    std::cout << "Histogram for '" << name << "' had " << num_mismatches << " mismatches" << std::endl;
}

void identify(cv::Mat const& img)
{
    using std::cout;
    using std::endl;
    cout << "Input img has:" << std::endl;
    cout << "- dims: " << img.dims << std::endl;
    cout << "- shape: " << img.cols << " cols x " << img.rows << " rows" << std::endl;
    cout << "- size: " << img.size[0] << ',' << img.size[1] << std::endl;
    cout << "- total: " << img.total() << std::endl;
    cout << "- type: " << (img.type() == CV_8UC3 ? "" : "not ") << "CV_8UC3" << std::endl;
    cout << "- elemSize: " << img.elemSize() << endl;
    cout << "- elemSize1: " << img.elemSize1() << endl;
    cout << "- isContinuous: " << img.isContinuous() << std::endl;
}

int main(int const argc, char const *const argv[])
{
    assert(argc == 2 && "Usage: cuda_histogram_test <img file>");

    // Load image into CPU
    cv::Mat cpu_img = cv::imread(argv[1]);
    identify(cpu_img);
    int32_t const img_size_bytes = cpu_img.rows * cpu_img.cols * 3;

    cudaError_t cuda_status = cudaSuccess;

    // Copy CPU image into CPU page-locked memory for async transfer
    uchar3 *cpu_img_pglocked = nullptr;
    cuda_status = cudaHostAlloc((void**) &cpu_img_pglocked, img_size_bytes, cudaHostAllocDefault);
    check(cuda_status, "Allocating page-locked CPU buffer for image");
    cuda_status = cudaMemcpy((void*) cpu_img_pglocked, (const void*) cpu_img.data, img_size_bytes, cudaMemcpyHostToHost);
    check(cuda_status, "Transferring image into CPU page-locked buffer");

    // Alloc GPU img memory
    uchar3 *gpu_img = nullptr;
    cuda_status = cudaMalloc((void**) &gpu_img, img_size_bytes);
    check(cuda_status, "cudaMalloc gpu-img");

    // Streams and events
    cudaStream_t cuda_stream[2];
    cuda_status = cudaStreamCreate(&cuda_stream[0]);
    check(cuda_status, "Create copy_stream #0");
    cuda_status = cudaStreamCreate(&cuda_stream[1]);
    check(cuda_status, "Create copy_stream #1");

    cudaEvent_t copy_complete_event;
    cuda_status = cudaEventCreate(&copy_complete_event);
    check(cuda_status, "Create copy-complete event");

    cudaEvent_t compute_complete_event[2];
    cuda_status = cudaEventCreate(&compute_complete_event[0]);
    check(cuda_status, "Create compute-complete event #0");
    cuda_status = cudaEventCreate(&compute_complete_event[1]);
    check(cuda_status, "Create compute-complete event #1");

    // CPU-side result-buffers
    int32_t *r_histogram_cpu[2]{nullptr, nullptr};
    int32_t *g_histogram_cpu[2]{nullptr, nullptr};
    int32_t *b_histogram_cpu[2]{nullptr, nullptr};
    for(int i = 0; i < 2; ++i) {
        std::string const iter_name = std::to_string(i);
        cuda_status = cudaHostAlloc((void**) &r_histogram_cpu[i], 256 * sizeof(int32_t), 0);
        check(cuda_status, "r_histogram_cpu host-alloc #" + iter_name);
        assert(r_histogram_cpu);
        cuda_status = cudaHostAlloc((void**) &g_histogram_cpu[i], 256 * sizeof(int32_t), 0);
        check(cuda_status, "g_histogram_cpu host-alloc #" + iter_name);
        assert(g_histogram_cpu);
        cuda_status = cudaHostAlloc((void**) &b_histogram_cpu[i], 256 * sizeof(int32_t), 0);
        check(cuda_status, "b_histogram_cpu host-alloc #" + iter_name);
        assert(b_histogram_cpu);
    }

    // GPU-side result buffers: 0 for global-mem-atomics kernel, 1 for shmem kernel
    int32_t *r_histogram_gpu[2]{nullptr, nullptr};
    int32_t *g_histogram_gpu[2]{nullptr, nullptr};
    int32_t *b_histogram_gpu[2]{nullptr, nullptr};
    for(int i = 0; i < 2; ++i) {
        std::string const iter_name = std::to_string(i);
        cuda_status = cudaMalloc((void**) &r_histogram_gpu[i], 256 * sizeof(int32_t));
        check(cuda_status, "R-histogram cudaMalloc #" + iter_name);
        cuda_status = cudaMalloc((void**) &g_histogram_gpu[i], 256 * sizeof(int32_t));
        check(cuda_status, "G-histogram cudaMalloc #" + iter_name);
        cuda_status = cudaMalloc((void**) &b_histogram_gpu[i], 256 * sizeof(int32_t));
        check(cuda_status, "B-histogram cudaMalloc #" + iter_name);
    }

    // CPU --> GPU image
    cuda_status = cudaMemcpyAsync((void*) gpu_img, (const void*) cpu_img_pglocked, img_size_bytes,
                                  cudaMemcpyHostToDevice, cuda_stream[0]);
    check(cuda_status, "CPU -> GPU image copy");
    cuda_status = cudaEventRecord(copy_complete_event, cuda_stream[0]);
    check(cuda_status, "Record copy-complete event");

    // Meanwhile, clear histograms
    resetHistograms<<<1,256,0,cuda_stream[0]>>>(r_histogram_gpu[0], g_histogram_gpu[0], b_histogram_gpu[0]);
    cuda_status = cudaGetLastError();
    check(cuda_status, "Launch resetHistograms kernel for global kernel");

    resetHistograms<<<1,256,0,cuda_stream[1]>>>(r_histogram_gpu[1], g_histogram_gpu[1], b_histogram_gpu[1]);
    cuda_status = cudaGetLastError();
    check(cuda_status, "Launch resetHistograms kernel for shmem kernel");

    // GPU-compute histograms after clearing them and transferring image
    CudaThreadConfig cuda_thread_cfg(cv::Size2i{cpu_img.cols, cpu_img.rows});

    Frame const frame{gpu_img, cpu_img.cols, cpu_img.rows};

    // global-atomics kernel in stream #0
    // In-stream implicit sync with cpu-->gpu copy
    rgbHistogramSinglePixGlobal<<<cuda_thread_cfg.grid, cuda_thread_cfg.block, 0, cuda_stream[0]>>>(
        frame,
        r_histogram_gpu[0],
        g_histogram_gpu[0],
        b_histogram_gpu[0]
    );
    cuda_status = cudaGetLastError();
    check(cuda_status, "Launch rgbHistogramSinglePixGlobal kernel");

    cuda_status = cudaEventRecord(compute_complete_event[0], cuda_stream[0]);
    check(cuda_status, "Record compute_complete_event #0");

    // shmem-atomics kernel in stream #1
    // need to sync with copy-complete in stream #0
    cuda_status = cudaStreamWaitEvent(cuda_stream[1], copy_complete_event, 0);
    check(cuda_status, "Registering wait for copy-complete event in stream #1");

    rgbHistogramSinglePixShMem<<<cuda_thread_cfg.grid, cuda_thread_cfg.block, 0, cuda_stream[1]>>>(
        frame,
        r_histogram_gpu[1],
        g_histogram_gpu[1],
        b_histogram_gpu[1]
    );
    cuda_status = cudaGetLastError();
    check(cuda_status, "Launch rgbHistogramSinglePixShMem kernel in copy kernel");

    cuda_status = cudaEventRecord(compute_complete_event[1], cuda_stream[1]);
    check(cuda_status, "Record compute_complete_event #1");

    // GPU --> CPU result copy
    // Intentionally make one stream copy the results of the other
    for(int i = 0; i < 2; ++i) {
        std::string const stream_name = "stream " + std::to_string(i);
        cuda_status = cudaStreamWaitEvent(cuda_stream[i], compute_complete_event[1-i], 0);
        check(cuda_status,
            "Registering wait for compute_complete_event for " + stream_name);

        cuda_status = cudaMemcpyAsync((void*) r_histogram_cpu[1-i], (const void*) r_histogram_gpu[1-i], 256 * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream[i]);
        check(cuda_status, "R-histogram GPU --> CPU copy by " + stream_name);
        cuda_status = cudaMemcpyAsync((void*) g_histogram_cpu[1-i], (const void*) g_histogram_gpu[1-i], 256 * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream[i]);
        check(cuda_status, "G-histogram GPU --> CPU copy by " + stream_name);
        cuda_status = cudaMemcpyAsync((void*) b_histogram_cpu[1-i], (const void*) b_histogram_gpu[1-i], 256 * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream[i]);
        check(cuda_status, "B-histogram GPU --> CPU copy by " + stream_name);
    }

    // Compute in CPU while kernels execute
    int32_t r_histogram_gold[256];
    int32_t g_histogram_gold[256];
    int32_t b_histogram_gold[256];
    rgbHistogramCpu(cpu_img, r_histogram_gold, g_histogram_gold, b_histogram_gold);

    // Compare CPU and GPU results after waiting for GPU
    for(int i = 0; i < 2; ++i) {
        std::string const stream_name = "stream " + std::to_string(i);
        cuda_status = cudaStreamSynchronize(cuda_stream[i]);
        check(cuda_status, "End-sync on " + stream_name);

        compareHistograms(r_histogram_cpu[i], r_histogram_gold, "R" + std::to_string(i));
        compareHistograms(g_histogram_cpu[i], g_histogram_gold, "G" + std::to_string(i));
        compareHistograms(b_histogram_cpu[i], b_histogram_gold, "B" + std::to_string(i));
    }

    // Release resources
    for(int i = 0; i < 2; ++i) {
        std::string const iter_name = std::to_string(i);
        cuda_status = cudaFreeHost(r_histogram_cpu[i]);
        check(cuda_status, "cuda free r_histogram_cpu #" + iter_name);
        cuda_status = cudaFreeHost(g_histogram_cpu[i]);
        check(cuda_status, "cuda free g_histogram_cpu #" + iter_name);
        cuda_status = cudaFreeHost(b_histogram_cpu[i]);
        check(cuda_status, "cuda free b_histogram_cpu #" + iter_name);

        cuda_status = cudaFree(r_histogram_gpu[i]);
        check(cuda_status, "cuda free r_histogram_gpu #" + iter_name);
        cuda_status = cudaFree(g_histogram_gpu[i]);
        check(cuda_status, "cuda free g_histogram_gpu #" + iter_name);
        cuda_status = cudaFree(b_histogram_gpu[i]);
        check(cuda_status, "cuda free b_histogram_gpu #" + iter_name);

        cuda_status = cudaEventDestroy(compute_complete_event[i]);
        check(cuda_status, "cuda destroy compute_complete_event #" + iter_name);

        cuda_status = cudaStreamDestroy(cuda_stream[i]);
        check(cuda_status, "cuda destroy stream #" + iter_name);
    }

    cuda_status = cudaEventDestroy(copy_complete_event);
    check(cuda_status, "cuda destroy copy_complete_event");

    cuda_status = cudaFree((void*) gpu_img);
    check(cuda_status, "cuda free gpu_img");

    cuda_status = cudaFreeHost((void*) cpu_img_pglocked);
    check(cuda_status, "cuda free CPU page-locked buffer");

    return 0;
}