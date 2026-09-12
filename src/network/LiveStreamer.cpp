#include "fall-detection/network/LiveStreamer.hpp"
#include "fall-detection/utils/SysLogger.hpp"
#include <cstdio>

namespace fall_detection
{
    namespace network
    {
        LiveStreamer::LiveStreamer(int width, int height, int fps) : width_(width), height_(height), fps_(fps) {}

        LiveStreamer::~LiveStreamer()
        {
            stop();
        }

        bool LiveStreamer::start(const std::string& rtmpUrl)
        {
            if (isStreaming_)
            {
                return true;
            }

            rtmpUrl_ = rtmpUrl;
            isStreaming_ = true;
            workerThread_ = std::thread(&LiveStreamer::streamLoop, this);
            LOG_INFO("[LiveStreamer] 推流线程已启动，目标 RTMP URL: {}", rtmpUrl_);
            return true;
        }

        void LiveStreamer::stop()
        {
            if (isStreaming_)
            {
                isStreaming_ = false;

                // 压入一张空图唤醒可能在阻塞等待的队列
                liveQueue_.push(cv::Mat());
                if (workerThread_.joinable())
                {
                    workerThread_.join();
                }
                LOG_INFO("[LiveStreamer] 推流线程已停止");
            }
        }

        void LiveStreamer::pushFrame(const cv::Mat& frame)
        {
            if (isStreaming_ && !frame.empty())
            {
                liveQueue_.push(frame);
            }
        }

        void LiveStreamer::streamLoop()
        {
            // 构建 FFmpeg 命令行
            // -f rawideo: 接收原始像素数据
            // -c:v h264_rkmpp: 使用 RKNN 硬件加速的 H.264 编码器
            std::string ffmpegCmd = "ffmpeg -y -nostats "
                "-f rawvideo -framerate 30 -vcodec rawvideo -pix_fmt bgr24 "
                "-s " + std::to_string(width_) + "x" + std::to_string(height_) + " "
                "-use_wallclock_as_timestamps 1 -i - "
                "-c:v h264_rkmpp -b:v 1000k -profile:v main -g 30 "
                "-f flv " + rtmpUrl_;

            // 打开 FFmpeg 进程的管道
            FILE* pipe = popen(ffmpegCmd.c_str(), "w");
            if (!pipe)
            {
                LOG_ERROR("无法启动 FFmpeg 推流管道！");
                isStreaming_ = false;
                return;
            }

            cv::Mat frame;
            while (isStreaming_)
            {
                // 从队列获取图像，带超时防止死锁
                if (liveQueue_.wait_for_and_pop(frame, std::chrono::milliseconds(500)))
                {
                    if (frame.empty()) continue;

                    // 确保尺寸一致
                    if (frame.cols != width_ || frame.rows != height_)
                    {
                        cv::resize(frame, frame, cv::Size(width_, height_));
                    }

                    // 将原始 BGR 像素数据直接写入 FFmpeg 管道
                    size_t written = fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe);
                    if (written != frame.total() * frame.elemSize())
                    {
                        LOG_ERROR("写入 FFmpeg 管道失败！");
                        break;
                    }
                }
            }

            pclose(pipe);
            isStreaming_ = false;
            LOG_INFO("[LiveStreamer] 推流线程退出");
        }
    }
}