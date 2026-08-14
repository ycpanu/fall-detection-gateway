#include "fall-detection/vision/CameraStreamer.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    CameraStreamer::CameraStreamer(int deviceId, concurrency::ThreadSafeQueue<cv::Mat>& frameQueue) : deviceId_(deviceId), frameQueue_(frameQueue), isRunning_(false) {}

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

        // 尝试打开摄像头设备
        capture_.open(deviceId_, cv::CAP_V4L2);
        if (!capture_.isOpened())
        {
            LOG_ERROR("无法打开摄像头设备号：{}", deviceId_);
            return false;
        }

        // 也可以在这里设置摄像头的分辨率，例如：
        // capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        // capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
        LOG_INFO("摄像头设备号 {} 已成功打开，准备启动采集线程...", deviceId_);

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

        while (isRunning_)
        {
            // 从硬件读取每一帧画面
            capture_ >> frame;

            if (frame.empty())
            {
                LOG_WARN("抓取到空帧，摄像头可能断开连接或出现异常.");

                // 短暂休眠防止 CPU 满载，并尝试重连/读取
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            // 将最新画面压入队列
            frameQueue_.push(frame);

            // 轻微休眠，释放部分 CPU 调度权（例如限制在 30fps 左右）
            std::this_thread::sleep_for(std::chrono::milliseconds(30));

        }
    }
}