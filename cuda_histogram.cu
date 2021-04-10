#include "cuda_histogram.cuh"

__global__ void resetHistograms(int32_t r_histogram[256],
                                int32_t g_histogram[256],
                                int32_t b_histogram[256])
{
    r_histogram[threadIdx.x] = 0;
    g_histogram[threadIdx.x] = 0;
    b_histogram[threadIdx.x] = 0;
#if 0
    int const grid_width = blockDim.x * gridDim.x;
    int const global_tid = (blockIdx.y * blockDim.y + threadIdx.y) * grid_width + blockIdx.x * blockDim.x + threadIdx.x;
    int const grid_height = blockDim.y * gridDim.y;
    int const num_kernel_threads = grid_width * grid_height;

    int bin_id = global_tid;
    while(bin_id < 256) {
        r_histogram[bin_id] = 0;
        g_histogram[bin_id] = 0;
        b_histogram[bin_id] = 0;
        bin_id += num_kernel_threads;
    }
#endif
}

__global__ void rgbHistogramSinglePixGlobal(Frame const frame,
                                            int32_t r_histogram[256],
                                            int32_t g_histogram[256],
                                            int32_t b_histogram[256])
{
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= 0 && x < frame.width && y >= 0 && y < frame.height) {
        int const pixel_linear_offset = y * frame.width + x;
        uchar3 const pixel = frame.ptr[pixel_linear_offset];
        (void) atomicAdd(r_histogram + pixel.x, 1);
        (void) atomicAdd(g_histogram + pixel.y, 1);
        (void) atomicAdd(b_histogram + pixel.z, 1);
    }
}

/// Grid should cover entire image
/// Call resetHistograms() before this in same stream
__global__ void rgbHistogramSinglePixShMem(Frame const frame,
                                           int32_t r_global[256],
                                           int32_t g_global[256],
                                           int32_t b_global[256])
{
    __shared__ int32_t r_shmem[256], g_shmem[256], b_shmem[256];

    int const block_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int const num_block_threads = blockDim.x * blockDim.y;

    // clear shared mem histogram per block

    int bin_id = block_tid;
    while(bin_id < 256) {
        // clear shared
        r_shmem[bin_id] = 0;
        g_shmem[bin_id] = 0;
        b_shmem[bin_id] = 0;
        bin_id += num_block_threads;
    }
    __syncthreads();

    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < frame.width && y < frame.height) {
        int const pixel_linear_offset = y * frame.width + x;
        uchar3 const pixel = frame.ptr[pixel_linear_offset];
        (void) atomicAdd(&r_shmem[pixel.x], 1);
        (void) atomicAdd(&g_shmem[pixel.y], 1);
        (void) atomicAdd(&b_shmem[pixel.z], 1);
    }
    __syncthreads();

    bin_id = block_tid;
    while(bin_id < 256) {
        (void) atomicAdd(&r_global[bin_id], r_shmem[bin_id]);
        (void) atomicAdd(&g_global[bin_id], g_shmem[bin_id]);
        (void) atomicAdd(&b_global[bin_id], b_shmem[bin_id]);
        bin_id += num_block_threads;
    }
}

constexpr uchar3 kBlack{0x00u, 0x00u, 0x00u};
constexpr uchar3 kWhite{0xFFu, 0xFFu, 0xFFu};
constexpr uchar3 kRed{0xFFu, 0x00u, 0x00u};
constexpr uchar3 kGreen{0x00u, 0xFFu, 0x00u};
constexpr uchar3 kBlue{0x00u, 0x00u, 0xFFu};

__device__
void drawHorizontalLine(Frame const frame,
                        int32_t x,
                        int32_t const y,
                        int32_t const length,
                        uchar3 const color)
{
    if(y >= 0 && y < frame.height) {
        int32_t const grid_width = gridDim.x * blockDim.x;
        int32_t rel_pix_id = blockIdx.x * blockDim.x + threadIdx.x;
        x += rel_pix_id;
        uchar3 *pixel = frame.ptr + y * frame.width + x;
        while(rel_pix_id < length) {
            if(x >= 0 && x < frame.width) {
                *pixel = color;
                pixel += grid_width;
                rel_pix_id += grid_width;
                x += grid_width;
            }
        }
    }
}

__device__
void drawHistogramTopBottomBorders(Frame const frame,
                                   int32_t const x0,
                                   int32_t const y0,
                                   uchar3 const color,
                                   int32_t const hist_width,
                                   int32_t const hist_height)
{
    drawHorizontalLine(frame, x0, y0, hist_width, color);
    drawHorizontalLine(frame, x0, y0 + hist_height - 1, hist_width, color);
}

// Call with >= 256 threads
__device__
void drawHistogramSingleColor(Frame const frame,
                              float const *const __restrict__ histogram,
                              int32_t const x0,
                              int32_t const y0,
                              int32_t const hist_width_per_bin,
                              int32_t const hist_height,
                              uchar3 const color,
                              uchar3 const background_color,
                              uchar3 const border_color = kWhite)
{
    int const hist_width = hist_width_per_bin * 256;
    drawHistogramTopBottomBorders(frame, x0, y0, border_color, hist_width, hist_height);

    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= 256) {
        return;
    }

    int const x_end = x0 + hist_width - 1;

    // histogram: row-by-row
    for(int y = (y0 + hist_height - 1); y > y0; --y) {
        if(y < 0 || y >= frame.height) { continue; }

        int x = x0 + tid * hist_width_per_bin;
        uchar3 *pixel = frame.ptr + y * frame.width + x;

        uchar3 pixel_color = background_color;
        if(x == x0 || x == x_end) { // left and right white border
            pixel_color = border_color;

        } else { // interior of histogram
            int32_t const hist_val_scaled_to_pix = (int32_t) (hist_height - 1) * histogram[tid];
            int32_t const height = (hist_height - 1) - (y - y0);
            pixel_color = (height <= hist_val_scaled_to_pix) ? color : background_color;
        }

        for(int32_t i_per_bin = 0; i_per_bin < hist_width_per_bin; ++i_per_bin) {
            if(x >= 0 && x < frame.width) {
                *pixel = pixel_color;
            }
            ++x;
            ++pixel;
        }

        pixel -= frame.width;
    } // for y
}

/// Call with one-dimensional block/grid with num threads >= 258
__global__
void drawHistogram(Frame const frame,
                   int32_t const x0,
                   int32_t const y0,
                   int32_t const hist_width_per_bin,
                   int32_t const hist_height,
                   float const r_histogram[256],
                   float const g_histogram[256],
                   float const b_histogram[256])
{
    drawHistogramSingleColor(frame, r_histogram, x0    , y0, hist_width_per_bin, hist_height, kRed, kBlack);
    drawHistogramSingleColor(frame, g_histogram, x0+300, y0, hist_width_per_bin, hist_height, kGreen, kBlack);
    drawHistogramSingleColor(frame, b_histogram, x0+600, y0, hist_width_per_bin, hist_height, kBlue, kBlack);
}
