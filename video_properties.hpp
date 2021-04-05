#ifndef VIDEO_PROPERTIES_HPP
#define VIDEO_PROPERTIES_HPP

#include "opencv2/videoio.hpp"

class VideoProperties
{
  public:
    explicit VideoProperties(cv::VideoCapture const& video_file) {
        num_frames_ = int(video_file.get(cv::CAP_PROP_FRAME_COUNT));
        frame_size_.width = int(video_file.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_size_.height = int(video_file.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_ = video_file.get(cv::CAP_PROP_FPS);
        cv_mat_type_ = int(video_file.get(cv::CAP_PROP_FORMAT));
        is_color_ = true;
    }

    int numFrames() const { return num_frames_; }
    cv::Size frameSize() const { return frame_size_; }
    double fps() const { return fps_; }
    bool isColor() const { return is_color_; }
    int cvMatType() const { return cv_mat_type_; }

  private:
    int num_frames_;
    cv::Size frame_size_;
    double fps_;
    bool is_color_;
    int cv_mat_type_;
};

#endif // VIDEO_PROPERTIES_HPP
