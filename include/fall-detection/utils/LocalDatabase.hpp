#pragma once

#include <string>
#include <vector>
#include <sqlite3.h>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include "fall-detection/event/AlertEvent.hpp"

namespace fall_detection
{
    namespace utils
    {
        struct DBAlertEvent : public event::AlertEvent
        {
            int dbId = 0;
        };

        struct PendingVideoUpload
        {
            std::string eventId;
            std::string videoPath;
            int retryCount = 0;
        };

        class LocalDatabase
        {
        public:
            LocalDatabase(const std::string& dbPath = "fall_detection.db");
            ~LocalDatabase();

            bool init();

            bool saveAlert(const event::AlertEvent& event);

            std::vector<DBAlertEvent> getPendingAlerts();

            bool markAsUploaded(const std::string& eventId);

            bool markVideoPending(
                const std::string& eventId,
                const std::string& videoPath
            );

            std::vector<PendingVideoUpload> getPendingVideos();

            bool markVideoUploaded(
                const std::string& eventId
            );

            bool incrementVideoRetry(
                const std::string& eventId
            );

        private:
            bool executeSQL(const std::string& sql);
            bool executeQueuedSQL(std::string sql);
            void dbWorkerLoop();

            struct SqlTask
            {
                std::string sql;
                std::shared_ptr<std::promise<bool>> completion;
            };

        private:
            std::string dbPath_;
            sqlite3* db_;
            std::mutex dbMtx_;
            std::atomic<bool> isInitialized_{false};

            std::queue<SqlTask> sqlQueue_;
            std::mutex queueMtx_;
            std::condition_variable cv_;
            std::thread workerThread_;
            std::atomic<bool> isRunning_{false};
        };
    }
}
