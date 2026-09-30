#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fall_detection
{
    namespace vision
    {

        struct VideoTask
        {
            std::deque<cv::Mat> frames;
            std::string eventId;
            std::string outputPath;
            int fps = 30;
        };

        struct PendingVideoRecord
        {
            std::deque<cv::Mat> frames;
            std::string eventId;
            std::string outputPath;
            int fps = 30;
            int remainingPostFrames = 0;
        };

        class VideoCacher
        {
        public:
            using VideoReadyCallback =
                std::function<void(
                    const std::string& eventId,
                    const std::string& outputPath
                )>;

            explicit VideoCacher(int maxFrames = 90);
            ~VideoCacher();

            void pushFrame(const cv::Mat& frame);

            void saveVideoAsync(
                const std::string& eventId,
                const std::string& outputPath,
                int fps = 30,
                int postFrames = 0
            );

            void setVideoReadyCallback(VideoReadyCallback callback);

        private:
            void encodingWorkerLoop();

        private:
            int maxFrame_;

            std::deque<cv::Mat> buffer_;
            std::mutex bufferMtx_;

            std::queue<VideoTask> taskQueue_;
            std::mutex queueMtx_;
            std::condition_variable cv_;

            std::thread workerThread_;
            std::atomic<bool> isRunning_{false};

            std::vector<PendingVideoRecord> pendingRecords_;

            VideoReadyCallback videoReadyCallback_;
        };

    }
}