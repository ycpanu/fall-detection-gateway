#include "fall-detection/vision/CameraStreamer.hpp"
#include "fall-detection/utils/SysLogger.hpp"
#include <chrono>

namespace fall_detection
{
    namespace vision
    {
        // 构造函数一：硬件摄像头模式
        CameraStreamer::CameraStreamer(int deviceId, concurrency::ThreadSafeQueue<cv::Mat>& frameQueue, VideoCacher& videoCacher) :
            isVideoFile_(false),
            deviceId_(deviceId),
            videoCacher_(videoCacher),
            frameQueue_(frameQueue),
            isRunning_(false)
        {}

        // 构造函数二：本地视频文件模式
        CameraStreamer::CameraStreamer(const std::string& videoPath, concurrency::ThreadSafeQueue<cv::Mat>& frameQueue, VideoCacher& videoCacher) :
            isVideoFile_(true),
            videoPath_(videoPath),
            videoCacher_(videoCacher),
            frameQueue_(frameQueue),
            isRunning_(false)
        {}

        CameraStreamer::~CameraStreamer()
        {
            stop();
        }

        bool CameraStreamer::start()
        {
            if (isRunning_)
            {
                LOG_WARN("视频采集线程已经在运行中，无需重复启动。");
                return true;
            }

            // 模式分流
            if (isVideoFile_)
            {
                capture_.open(videoPath_);
                if (!capture_.isOpened())
                {
                    LOG_ERROR("无法打开视频文件：{}", videoPath_);
                    return false;
                }
                LOG_INFO("视频文件 [{}] 已成功打开，启动模拟回归测试...", videoPath_);
            }
            else
            {
                // 强制指定 V4L2 驱动
                capture_.open(deviceId_, cv::CAP_V4L2);
                if (!capture_.isOpened())
                {
                    LOG_ERROR("无法打开摄像头：{}", deviceId_);
                    return false;
                }

                // 锁定 MJPG 格式，解决部分 V4L2 驱动的抓图死锁问题；
                // 分辨率不再强制锁定，由摄像头输出原生分辨率，模型侧通过 letterbox 保持比例
                capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                LOG_INFO("摄像头 [{}] 已成功打开，准备启动实时采集线程...", deviceId_);
            }

            // 标记运行状态并启动独立线程
            isRunning_ = true;
            workerThread_ = std::thread(&CameraStreamer::captureLoop, this);

            return true;
        }

        void CameraStreamer::stop()
        {
            if (isRunning_)
            {
                LOG_INFO("正在停止视频采集线程...");
                isRunning_ = false;

                //等待工作线程安全退出
                if (workerThread_.joinable())
                {
                    workerThread_.join();
                }

                // 释放摄像头硬件资源
                if (capture_.isOpened())
                {
                    capture_.release();
                }

                LOG_INFO("视频采集线程已安全停止，摄像头硬件已释放。");
            }
        }

        void CameraStreamer::captureLoop()
        {
            cv::Mat frame;

            // 动态计算休眠间隔
            // 视频：按原生 FPS 投递，摄像头：默认限制在约 33ms(30FPS)
            int delayMs = 33;
            if (isVideoFile_)
            {
                double fps = capture_.get(cv::CAP_PROP_FPS);
                if (fps > 0)
                {
                    delayMs = static_cast<int>(1000.0 / fps);
                }
            }

            while (isRunning_)
            {
                // 从硬件读取每一帧画面
                capture_ >> frame;

                if (frame.empty())
                {
                    if (isVideoFile_)
                    {
                        // 视频播放到末尾时，自动重置帧指针实现循环播放
                        LOG_INFO("视频播放完毕，已自动循环播放");
                        capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
                        continue;
                    }
                    else
                    {
                        LOG_WARN("抓取到空帧，摄像头可能断开连接或出现异常.");

                        // 短暂休眠防止 CPU 满载，并尝试重连/读取
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }
                    
                }

                // 将最新画面压入队列
                frameQueue_.push(frame);
                videoCacher_.pushFrame(frame);
                // 轻微休眠，释放部分 CPU 调度权（例如限制在 30fps 左右）
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
    }
}