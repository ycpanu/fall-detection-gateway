#include "fall-detection/network/VideoUploader.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fall_detection
{
    namespace network
    {

        VideoUploader::VideoUploader(
            utils::LocalDatabase& database,
            std::string serverUrl,
            int retryIntervalSeconds)
            : database_(database),
            serverUrl_(std::move(serverUrl)),
            retryIntervalSeconds_(retryIntervalSeconds)
        {
            while (!serverUrl_.empty() && serverUrl_.back() == '/')
                serverUrl_.pop_back();
        }

        VideoUploader::~VideoUploader()
        {
            stop();
        }

        void VideoUploader::start()
        {
            bool expected = false;

            if (!running_.compare_exchange_strong(expected, true))
                return;

            workerThread_ = std::thread(
                &VideoUploader::workerLoop,
                this
            );

            LOG_INFO(
                "[VideoUpload] 视频上传线程已启动，server={}",
                serverUrl_
            );
        }

        void VideoUploader::stop()
        {
            if (!running_.exchange(false))
                return;

            cv_.notify_all();

            if (workerThread_.joinable())
                workerThread_.join();

            LOG_INFO("[VideoUpload] 视频上传线程已退出");
        }

        void VideoUploader::notifyNewTask()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                wakeRequested_ = true;
            }
            cv_.notify_one();
        }

        void VideoUploader::setOnline(bool online)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const bool wasOnline = online_.exchange(online);
                if (online && !wasOnline)
                    ++connectionGeneration_;
                wakeRequested_ = true;
            }
            cv_.notify_one();
        }

        std::string VideoUploader::shellQuote(const std::string& value)
        {
            std::string result = "'";

            for (char ch : value)
            {
                if (ch == '\'')
                    result += "'\"'\"'";
                else
                    result += ch;
            }

            result += "'";
            return result;
        }

        bool VideoUploader::uploadVideo(
            const utils::PendingVideoUpload& video)
        {
            if (!std::filesystem::exists(video.videoPath))
            {
                LOG_WARN(
                    "[VideoUpload] 本地视频不存在：event_id={}, path={}",
                    video.eventId,
                    video.videoPath
                );

                return false;
            }

            const std::string uploadUrl =
                serverUrl_ + "/api/upload/video";

            const std::string formValue =
                "file=@" + video.videoPath;

            const std::string command =
                "curl -sS --fail "
                "--connect-timeout 5 "
                "--max-time 60 "
                "-X POST "
                "-F " + shellQuote(formValue) + " " +
                shellQuote(uploadUrl) +
                " >/dev/null";

            LOG_INFO(
                "[VideoUpload] 开始上传：event_id={}, path={}, retry={}",
                video.eventId,
                video.videoPath,
                video.retryCount
            );

            const int result = std::system(command.c_str());

            if (result != 0)
            {
                LOG_WARN(
                    "[VideoUpload] 上传失败：event_id={}, curl_result={}",
                    video.eventId,
                    result
                );

                return false;
            }

            LOG_INFO(
                "[VideoUpload] 上传成功：event_id={}",
                video.eventId
            );

            return true;
        }

        void VideoUploader::workerLoop()
        {
            using Clock = std::chrono::steady_clock;
            std::unordered_map<std::string, Clock::time_point> nextAttempt;
            std::unordered_map<std::string, unsigned int> failures;
            unsigned long seenGeneration = 0;

            while (running_)
            {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() { return !running_ || online_; });
                    if (!running_) break;
                    if (seenGeneration != connectionGeneration_)
                    {
                        nextAttempt.clear();
                        failures.clear();
                        seenGeneration = connectionGeneration_;
                    }
                    wakeRequested_ = false;
                }

                const auto pendingVideos =
                    database_.getPendingVideos();

                std::unordered_set<std::string> pendingIds;
                auto nextWake = Clock::now() + std::chrono::minutes(5);

                for (const auto& video : pendingVideos)
                {
                    pendingIds.insert(video.eventId);
                    if (!running_ || !online_)
                        break;

                    const auto now = Clock::now();
                    const auto due = nextAttempt.find(video.eventId);
                    if (due != nextAttempt.end() && now < due->second)
                    {
                        nextWake = std::min(nextWake, due->second);
                        continue;
                    }

                    if (uploadVideo(video))
                    {
                        if (database_.markVideoUploaded(video.eventId))
                        {
                            nextAttempt.erase(video.eventId);
                            failures.erase(video.eventId);
                            continue;
                        }
                    }
                    else
                    {
                        database_.incrementVideoRetry(video.eventId);
                    }

                    auto& failureCount = failures[video.eventId];
                    failureCount = std::min(failureCount + 1, 6u);
                    const auto delay = std::chrono::seconds(
                        std::min(retryIntervalSeconds_ * (1 << (failureCount - 1)), 300));
                    nextAttempt[video.eventId] = Clock::now() + delay;
                    nextWake = std::min(nextWake, nextAttempt[video.eventId]);
                }

                for (auto it = nextAttempt.begin(); it != nextAttempt.end();)
                {
                    if (!pendingIds.count(it->first))
                    {
                        failures.erase(it->first);
                        it = nextAttempt.erase(it);
                    }
                    else ++it;
                }

                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_until(lock, nextWake, [this, seenGeneration]()
                {
                    return !running_ || !online_ || wakeRequested_ ||
                        connectionGeneration_ != seenGeneration;
                });
            }
        }

    }
}
