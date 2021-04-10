#include "cuda.h"

__device__ __forceinline__ int sqr(int const x) { return (x * x); }

// Call with 2D thread-organization. Could be called with 1D arrangement but this is for exercise.
__global__ void clearFrame(uchar3 *const frame, int const frame_width, int const frame_height)
{
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < frame_width && y < frame_height) {
        int const row_size = gridDim.x * blockDim.x;
        int const offset = y * row_size + x;
        frame[offset] = make_uchar3(0u,0u,0u);
    }
}

__global__ void drawCircle(int const x0, int const y0, int const radius, uchar3 const color,
                           uchar3 *const frame, int const frame_width, int const frame_height)
{
    int const tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int const tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int const x = x0 + (tid_x - radius);
    int const y = y0 + (tid_y - radius);
    if(x >= 0 && x < frame_width && y >= 0 && y < frame_height) {
        int const hypot_sqr = sqr(x - x0) + sqr(y - y0);
        if(hypot_sqr <= sqr(radius)) {
            uchar3 *const pixel = frame + y * frame_width + x;
            *pixel = color;
        }
    }
}

__global__ void drawRect(int const x0, int const y0,
                         int const x1, int const y1,
                         uchar3 const color,
                         uchar3 *const frame, int const frame_width, int const frame_height)
 {
     int const x = x0 + threadIdx.x;
     int const y = y0 + threadIdx.y;
     
     if(x >= 0 && x < frame_width && y >= 0 && y < frame_height) {
        uchar3 *const pixel = frame + y * frame_width + x;
        *pixel = color;
     }
 }
