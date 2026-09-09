#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <string>
#include "fall-detection/concurrency/ThreadSafeQueue.hpp"

namespace fall_detection
{
    namespace network
    {
        class LiveStreamer
        {
        public:
            LiveStreamer(int width = 640, int height = 480, int fps = 30);
            ~LiveStreamer();

            // 启动推流
            bool start(const std::string& rtmpUrl);

            // 停止推流
            void stop();

            // 压入最新画面，由 CameraCapture 线程调用
            void pushFrame(const cv::Mat& frame);

            // 是否正在推流
            bool isStreaming() const
            {
                return isStreaming_;
            }

        private:
            void streamLoop();

        private:
            int width_;
            int height_;
            int fps_;
            std::string rtmpUrl_;
            std::atomic<bool> isStreaming_{false};
            std::thread workerThread_;

            // 直播专用帧队列，容量小一点，保证实时性，队满自动丢帧
            concurrency::ThreadSafeQueue<cv::Mat> liveQueue_{3};
        };
    }
}