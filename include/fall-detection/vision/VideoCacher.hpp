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
            // 停止采集后完成排队中的编码，并等待回调写入数据库。
            void stopAndDrain();

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
