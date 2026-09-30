#include "fall-detection/network/VideoUploader.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
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
            while (running_)
            {
                const auto pendingVideos =
                    database_.getPendingVideos();

                for (const auto& video : pendingVideos)
                {
                    if (!running_)
                        break;

                    if (uploadVideo(video))
                    {
                        database_.markVideoUploaded(
                            video.eventId
                        );
                    }
                    else
                    {
                        database_.incrementVideoRetry(
                            video.eventId
                        );
                    }
                }

                std::unique_lock<std::mutex> lock(mutex_);

                cv_.wait_for(
                    lock,
                    std::chrono::seconds(retryIntervalSeconds_)
                );
            }
        }

    }
}