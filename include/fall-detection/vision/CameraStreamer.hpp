#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include "fall-detection/concurrency/ThreadSafeQueue.hpp"
#include "fall-detection/vision/VideoCacher.hpp"

namespace fall_detection
{
    namespace vision
    {
        /**
         * @brief 摄像头视频流采集类，运行在独立线程
         * 
         * 负责通过 OpenCV 极速抓取视频帧，并将其压入线程安全队列，
         * 避免因后端 AI 推理或网络波动导致画面采集卡顿。
         */
        class CameraStreamer
        {
            public:
                /**
                 * @brief 构造函数，模式一：USB 摄像头硬件驱动
                 * @param deviceId 摄像头设备号，0 代表 /dev/video0
                 * @param frameQueue 绑定的全局线程安全图像队列的引用
                 */
                CameraStreamer(int deviceId, concurrency::ThreadSafeQueue<cv::Mat>& frameQueue, VideoCacher& videoCacher);

                /**
                 * @brief 模式二：本地视频模拟
                 * @param videoPath 视频文件路径
                 * @param frameQueue 绑定的全局线程安全图像队列
                 */
                CameraStreamer(const std::string& videoPath, concurrency::ThreadSafeQueue<cv::Mat>& frameQueue, VideoCacher& videoCacher);
                ~CameraStreamer();

                // @brief 启动视频采集线程
                bool start();

                // @brief 停止视频采集线程并释放摄像头
                void stop();

            private:
                // @brief 内部工作线程的核心死循环函数
                void captureLoop();

            private:
                bool isVideoFile_;                      // 模式标志
                int deviceId_;                          // 摄像头设备号
                std::string videoPath_;                 // 文件模式下的视频路径
                VideoCacher& videoCacher_;
                cv::VideoCapture capture_;              // OpenCV 视频捕获对象
                concurrency::ThreadSafeQueue<cv::Mat>& frameQueue_;  // 引用外部图像队列
                std::thread workerThread_;              // 独立工作线程
                std::atomic<bool> isRunning_;           // 线程运行状态标志位
        };
    }
}