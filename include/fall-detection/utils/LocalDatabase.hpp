#pragma once

#include <string>
#include <vector>
#include <sqlite3.h>
#include "fall-detection/vision/FallRuleEngine.hpp"

namespace fall_detection
{
    namespace utils
    {
        /**
         * @brief 继承并扩展 AlertEvent，增加数据库记录的唯一 ID 字段，
         * 用于物联网续传成功后，精准地去数据库里将这条记录标记为“已上传”。
         */
        struct DBAlertEvent : public vision::AlertEvent
        {
            int dbId;
        };

        /**
         * @brief 本地SQLite 数据库管理器
         * 负责断网时，将报警数据持久化到本地磁盘
         * 网络恢复后，提取积压地报警数据供通信层上传
         */
        class LocalDatabase
        {
            public:
                LocalDatabase(const std::string& dbPath = "fall_detection.db");

                ~LocalDatabase();

                bool init();

                bool saveAlert(const vision::AlertEvent& event);

                std::vector<DBAlertEvent> getPendingAlerts();

                bool markAsUploaded(int id);
            
            private:
                bool executeSQL(const std::string& sql);

            private:
                std::string dbPath_;
                sqlite3* db_;
                bool isInitialized_;
        };
    }
}