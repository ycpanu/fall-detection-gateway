#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <queue>
#include <mutex>
#include <string>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace fall_detection
{
    namespace vision
    {
        // 视频编码异步任务包
        struct VideoTask 
        {
            std::deque<cv::Mat> frames;
            std::string outputPath;
            int fps;
        };

        class VideoCacher
        {
            public:
                VideoCacher(int maxFrames = 90);
                ~VideoCacher();

                void pushFrame(const cv::Mat& frame);
                void saveVideoAsync(const std::string& outputPath, int fps = 30);
            
            private:
                // 常驻后台编码工作线程
                void encodingWorkerLoop();

            private:
                int maxFrame_;
                std::deque<cv::Mat> buffer_;
                std::mutex bufferMtx_;

                // 异步任务队列与线程同步原语
                std::queue<VideoTask> taskQueue_;
                std::mutex queueMtx_;
                std::condition_variable cv_;
                std::thread workerThread_;
                std::atomic<bool> isRunning_{false};
        };
    }
}