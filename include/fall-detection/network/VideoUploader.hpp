#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "fall-detection/utils/LocalDatabase.hpp"

namespace fall_detection
{
namespace network
{

class VideoUploader
{
public:
    VideoUploader(
        utils::LocalDatabase& database,
        std::string serverUrl,
        int retryIntervalSeconds = 5
    );

    ~VideoUploader();

    void start();
    void stop();

    // 新视频进入 PENDING 后立即唤醒线程，
    // 不需要等下一次定时检查。
    void notifyNewTask();

private:
    void workerLoop();
    bool uploadVideo(const utils::PendingVideoUpload& video);

    static std::string shellQuote(const std::string& value);

private:
    utils::LocalDatabase& database_;
    std::string serverUrl_;
    int retryIntervalSeconds_;

    std::atomic<bool> running_{false};
    std::thread workerThread_;

    std::mutex mutex_;
    std::condition_variable cv_;
};

}
}