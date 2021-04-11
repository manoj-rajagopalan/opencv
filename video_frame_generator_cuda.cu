#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

#include "shapes.hpp"
#include "moving_object.hpp"

#include "draw_shapes.cuh"

int constexpr FRAME_WIDTH = 960;
int constexpr FRAME_HEIGHT = 600;

using std::cout;
using std::endl;

void cudaCheckSuccess(cudaError_t const cuda_status, std::string const& message)
{
    if(cudaSuccess != cuda_status) {
        cout << "CUDA ERROR " << cuda_status << ": " << message << endl;
        cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << endl;
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

std::ostream& operator << (std::ostream& o, dim3 const& d) {
    o << '(' << d.x << ',' << d.y << ')';
    return o;
}

// Helper class to write images to disk in a separate CPU-worker-thread
class DiskWriteCtx final // because it composes std::thread which runs as soon as constructed
{
  public:
    DiskWriteCtx(cv::Mat const& mat,
                 cudaStream_t cuda_stream)
    : mat_(mat),
      cuda_stream_(cuda_stream),
      frame_id_(-1),
      join_(false),
      ctx_id_(ctx_id_counter_++),
      thread_(&DiskWriteCtx::run_, this)
    {}

    // Called by host
    void postFrame(int frame_id);
    void join();

  private:
    void run_();

    cv::Mat const& mat_;
    cudaStream_t cuda_stream_;
    std::atomic<int> frame_id_;
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> join_;
    int ctx_id_;

    // keep as last non-static member because it runs as soon as constructed, and
    // all others must be fully constructed before that
    std::thread thread_; 

    static int ctx_id_counter_ = 0;
};

void DiskWriteCtx::run_()
{
    cout << "Running write_ctx " << ctx_id_ << endl;
    while(true) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&](){ return join_ || frame_id_ >= 0; });
        if(join_) {
            break;
        }
        cout << "Writing frame " << frame_id_ << " in write_ctx " << ctx_id_ << endl; 
        cudaError_t cuda_status = cudaStreamSynchronize(cuda_stream);
        cudaCheckSuccess(cuda_status, "Error sync-ing to cuda stream for frame " + std::to_string(frame_id_));
        std::ostringstream s;
        s << prefix << '-' << std::setw(3) << std::setfill('0') << frame_id_ << ".png";
        cv::imwrite(s.str(), mat_);
        frame_id_ = -1;
        cv_.notify_one();
    }
}

void DiskWriteCtx::join()
{
    cout << "Joining write_ctx " << ctx_id_ << endl;
    join_ = true;
    cv_.notify_one(); // wake up sleeping worker
    thread_.join();
}

void DiskWriteCtx::postFrame(int frame_id)
{
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&](){ return frame_id_ < 0; });
    frame_id_ = frame_id;
    cout << "Posted frame " << frame_id << " to write_ctx " << ctx_id_ << endl;
}

void generateFrames(int const num_frames,
                    std::vector<MovingObject>& circles,
                    std::string const& prefix)
{
    cudaError_t cuda_status = cudaSuccess;

    uchar3 *cpu_frame[2] {nullptr, nullptr};
    cv::Mat frames[2];
    uchar3 *dev_frame[2] {nullptr, nullptr};
    cudaStream_t cuda_stream[2];
    int const frame_num_bytes = FRAME_HEIGHT * FRAME_WIDTH * 3;
    for(int i = 0; i < 2; ++i) {
        cuda_status = cudaMallocHost((void**) &cpu_frame[i], frame_num_bytes); // pinned
        cudaCheckSuccess(cuda_status, "Error host-mallocing cv::Mat " + std::to_string(i));
        frames[i] = cv::Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3, cpu_frame[i], /*step=*/FRAME_WIDTH * 3);
        cuda_status = cudaStreamCreate(&cuda_stream[i]);
        cudaCheckSuccess(cuda_status, "Error creating CUDA stream " + std::to_string(i));
        cuda_status = cudaMalloc((void**) &dev_frame[i], frame_num_bytes);
        cudaCheckSuccess(cuda_status, "Error allocating device-frame " + std::to_string(i));
    }

    // Spawns two worker threads
    DiskWriteCtx write_ctx[2]{
        DiskWriteCtx{frames[0], cuda_stream[0]},
        DiskWriteCtx{frames[1], cuda_stream[1]}
    };

    dim3 const clear_block(16,16,1);
    dim3 const clear_grid((FRAME_WIDTH+15)/16, (FRAME_HEIGHT+15)/16, 1);

    for(int i_frame = 0; i_frame < num_frames; ++i_frame) {
        std::string const frame_string = "frame " + std::to_string(i_frame);
        std::cout << "Drawing " << frame_string << '\n';

        int const stream_id = i_frame % 2;

        // Clear frame
        clearFrame<<<clear_grid, clear_block, 0, cuda_stream[stream_id]>>>(dev_frame, FRAME_WIDTH, FRAME_HEIGHT);
        cuda_status = cudaGetLastError();
        cudaCheckSuccess(cuda_status, "Error launching 'clear' kernel in " + frame_string);

        // Draw circles
        int circle_counter = 0;
        for(auto &obj : circles) {
            cv::Rect2i const bb = obj.boundingBox();
            uchar3 const color{obj.shape().color()[0], obj.shape().color()[1], obj.shape().color()[2]};
            cv::Point2i const position = obj.position();
            int const radius = bb.size().width / 2;
            CudaThreadConfig const cuda_thread_config(bb.size());
            std::cout << "- Circle " << circle_counter
                      << " bb = " << cuda_thread_config.bb_size
                      << " grid = " << cuda_thread_config.grid
                      << " block = " << cuda_thread_config.block
                      << " fragmentation = " << cuda_thread_config.fragmentation()
                      << '\n';
            drawCircle<<<cuda_thread_config.grid, cuda_thread_config.block, 0, cuda_stream[stream_id]>>>(position.x, position.y, radius, color, dev_frame, FRAME_WIDTH, FRAME_HEIGHT);
            cuda_status = cudaGetLastError();
            cudaCheckSuccess(cuda_status, "Error launching kernel for circle " + std::to_string(circle_counter) + " in " + frame_string);
            obj.update();
            ++circle_counter;
        }

        // Fetch frame from GPU and write to img file
        cuda_status = cudaMemcpyAsync((void*) frame.data, (void*) dev_frame, frame_num_bytes, cudaMemcpyDeviceToHost, stream_id);
        cudaCheckSuccess(cuda_status, "Error copying data out of GPU in " + frame_string);
        write_ctx.postFrame(i_frame); // to worker thread
    }

    // cleanup
    for(int i = 0; i < 2; ++i) {
        write_ctx[i].join(); // worker thread
        cuda_status = cudaFreeHost((void*) cpu_frame[i]);
        cudaCheckSuccess(cuda_status, "Error freeing cpu_frame " + std::to_string(i));
        cuda_status = cudaStreamDestroy(cuda_stream[i]);
        cudaCheckSuccess(cuda_status, "Error destroying cuda-stream " + std::to_string(i));
        cuda_status = cudaFree((void*) dev_frame[i]);
        cudaCheckSuccess(cuda_status, "Error freeing gpu-frame " + std::to_string(i));
    }
}

int main(int const argc, char const *argv[])
{
    assert(argc == 3 && "Usage: video_frame_generator <frames> <prefix>-nnn.png");
    int const num_frames = std::atoi(argv[1]);
    std::vector<MovingObject> circles = generateRandomMovingCircles(25, FRAME_WIDTH, FRAME_HEIGHT);
    generateFrames(num_frames, circles, argv[2]);
    return 0;
}
