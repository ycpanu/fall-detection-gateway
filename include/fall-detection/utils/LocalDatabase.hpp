#pragma once

#include <string>
#include <vector>
#include <sqlite3.h>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "fall-detection/vision/FallRuleEngine.hpp"

namespace fall_detection
{
    namespace utils
    {
        struct DBAlertEvent : public vision::AlertEvent
        {
            int dbId;
        };

        class LocalDatabase
        {
            public:
                LocalDatabase(const std::string& dbPath = "fall_detection.db");
                ~LocalDatabase();

                bool init();

                // 异步存盘，极速返回，绝对不阻塞调用线程
                bool saveAlert(const vision::AlertEvent& event);

                // 同步读取积压数据（由于仅在重连时调用，不影响实时主干，可保持同步）
                std::vector<DBAlertEvent> getPendingAlerts();

                // 异步更新数据库上传状态
                bool markAsUploaded(int id);
            
            private:
                // 同步执行底层的 SQL (仅限内部或后台线程调用)
                bool executeSQL(const std::string& sql);

                // 常驻后台的独立写盘线程
                void dbWorkerLoop();

            private:
                std::string dbPath_;
                sqlite3* db_;
                std::atomic<bool> isInitialized_{false};

                // 异步持久化任务队列与并发控制原语
                std::queue<std::string> sqlQueue_;
                std::mutex queueMtx_;
                std::condition_variable cv_;
                std::thread workerThread_;
                std::atomic<bool> isRunning_{false};
        };
    }
}