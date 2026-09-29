#include <sstream>

#include "fall-detection/utils/LocalDatabase.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace utils
    {
        LocalDatabase::LocalDatabase(const std::string& dbPath) : dbPath_(dbPath), db_(nullptr) {}

        LocalDatabase::~LocalDatabase()
        {
            // 1. 优雅停止后台写盘线程，确保关机前队列中的 SQL 被全部执行完毕
            if (isRunning_)
            {
                isRunning_ = false;
                cv_.notify_one();
                if (workerThread_.joinable())
                {
                    workerThread_.join();
                }
            }

            // 2. 释放数据库句柄
            if (db_ != nullptr)
            {
                sqlite3_close(db_);
                LOG_INFO("本地数据库连接已安全关闭！");
            }
        }

        bool LocalDatabase::init()
        {
            if (sqlite3_open(dbPath_.c_str(), &db_) != SQLITE_OK)
            {
                LOG_ERROR("无法打开本地数据库：{}", sqlite3_errmsg(db_));
                sqlite3_close(db_);
                db_ = nullptr;
                return false;
            }

            // 【核心优化点】开启 WAL (Write-Ahead Logging) 模式
            // 大幅提升 SQLite 的并发读写性能，将随机 I/O 转化为顺序 I/O
            executeSQL("PRAGMA journal_mode=WAL;");

            std::string createTableSQL =
                "CREATE TABLE IF NOT EXISTS alerts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "event_id TEXT NOT NULL UNIQUE, "
                "device_id TEXT NOT NULL, "
                "deployment_area TEXT DEFAULT '', "
                "event_type TEXT NOT NULL, "
                "sources TEXT NOT NULL, "
                "timestamp INTEGER NOT NULL, "
                "person_track_id INTEGER DEFAULT -1, "
                "trigger_x INTEGER DEFAULT 0, "
                "trigger_y INTEGER DEFAULT 0, "
                "keyword TEXT DEFAULT '', "
                "video_path TEXT DEFAULT '', "
                "event_status TEXT NOT NULL, "
                "upload_status TEXT NOT NULL DEFAULT 'pending'"
                ");";

            if (!executeSQL(createTableSQL))
            {
                LOG_ERROR("创建本地数据库表失败！");
                return false;
            }

            isInitialized_ = true;
            isRunning_ = true;
            
            // 启动单例常驻后台写盘守护线程
            workerThread_ = std::thread(&LocalDatabase::dbWorkerLoop, this);
            
            LOG_INFO("本地断网缓存数据库初始化成功！文件路径：{}", dbPath_);
            return true;
        }

        bool LocalDatabase::executeSQL(const std::string& sql)
        {
            char* errMsg = nullptr;
            if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK)
            {
                LOG_ERROR("SQL 执行错误：{} (SQL: {})", errMsg, sql);
                sqlite3_free(errMsg);
                return false;
            }
            return true;
        }

        bool LocalDatabase::saveAlert(const event::AlertEvent& alertEvent)
        {
            if (!isInitialized_)
            {
                return false;
            }

            // SQL 字符串转义，避免关键词、区域名等包含单引号时导致 SQL 语句失败
            auto escapeSql = [](const std::string& value)
            {
                std::string result;
                result.reserve(value.size());

                for (char ch : value)
                {
                    if (ch == '\'')
                    {
                        result += "''";
                    }
                    else
                    {
                        result += ch;
                    }
                }

                return result;
            };

            // ==========================================
            // 1. 报警来源序列化
            //
            // VISION
            // VOICE
            // VISION,VOICE
            // ==========================================
            std::string sourcesStr;

            for (std::size_t i = 0; i < alertEvent.source.size(); ++i)
            {
                if (i > 0)
                {
                    sourcesStr += ",";
                }

                sourcesStr += event::toString(alertEvent.source[i]);
            }

            // ==========================================
            // 2. 枚举转换为数据库字符串
            // ==========================================
            std::string eventTypeStr =
                event::toString(alertEvent.eventType);

            std::string eventStatusStr =
                event::toString(alertEvent.status);

            // ==========================================
            // 3. 对字符串字段进行 SQL 转义
            // ==========================================
            std::string eventId =
                escapeSql(alertEvent.eventId);

            std::string deviceId =
                escapeSql(alertEvent.deviceId);

            std::string deploymentArea =
                escapeSql(alertEvent.deploymentArea);

            std::string keyword =
                escapeSql(alertEvent.keyword);

            std::string videoPath =
                escapeSql(alertEvent.videoPath);

            sourcesStr =
                escapeSql(sourcesStr);

            eventTypeStr =
                escapeSql(eventTypeStr);

            eventStatusStr =
                escapeSql(eventStatusStr);

            // ==========================================
            // 4. INSERT + UPSERT
            //
            // 第一次 event_id：
            //      INSERT
            //
            // 相同 event_id 再次到达：
            //      UPDATE 融合信息
            //
            // event_type 和 timestamp 不更新，
            // 始终保留第一次报警的类型和时间。
            // ==========================================
            std::string insertSQL =
                "INSERT INTO alerts ("
                "event_id, "
                "device_id, "
                "deployment_area, "
                "event_type, "
                "sources, "
                "timestamp, "
                "person_track_id, "
                "trigger_x, "
                "trigger_y, "
                "keyword, "
                "video_path, "
                "event_status, "
                "upload_status"
                ") VALUES ('" +

                eventId + "', '" +
                deviceId + "', '" +
                deploymentArea + "', '" +
                eventTypeStr + "', '" +
                sourcesStr + "', " +

                std::to_string(alertEvent.timestamp) + ", " +
                std::to_string(alertEvent.personTrackId) + ", " +
                std::to_string(alertEvent.triggerBoxX) + ", " +
                std::to_string(alertEvent.triggerBoxY) + ", '" +

                keyword + "', '" +
                videoPath + "', '" +
                eventStatusStr + "', 'PENDING') "

                "ON CONFLICT(event_id) DO UPDATE SET "

                // 更新多模态来源
                "sources = excluded.sources, "

                // 如果后到的是视觉事件，则补充人员轨迹信息
                "person_track_id = CASE "
                "WHEN excluded.person_track_id >= 0 "
                "THEN excluded.person_track_id "
                "ELSE alerts.person_track_id "
                "END, "

                // 如果存在视觉信息，则更新触发位置
                "trigger_x = CASE "
                "WHEN excluded.person_track_id >= 0 "
                "THEN excluded.trigger_x "
                "ELSE alerts.trigger_x "
                "END, "

                "trigger_y = CASE "
                "WHEN excluded.person_track_id >= 0 "
                "THEN excluded.trigger_y "
                "ELSE alerts.trigger_y "
                "END, "

                // 如果后到的是语音事件，则补充关键词
                "keyword = CASE "
                "WHEN excluded.keyword <> '' "
                "THEN excluded.keyword "
                "ELSE alerts.keyword "
                "END, "

                // 如果后续视觉事件生成了录像，则补充视频路径
                "video_path = CASE "
                "WHEN excluded.video_path <> '' "
                "THEN excluded.video_path "
                "ELSE alerts.video_path "
                "END, "

                // 部署区域以当前配置为准
                "deployment_area = excluded.deployment_area, "

                // 数据发生更新，需要重新同步云端
                "upload_status = 'PENDING';";

            // 5. 放入 SQLite 后台写入队列
            {
                std::lock_guard<std::mutex> lock(queueMtx_);
                sqlQueue_.push(std::move(insertSQL));
            }

            cv_.notify_one();

            LOG_WARN(
                "报警事件已进入本地持久化队列："
                "event_id={}, type={}, sources={}",
                alertEvent.eventId,
                event::toString(alertEvent.eventType),
                sourcesStr
            );

            return true;
        }

        std::vector<DBAlertEvent>LocalDatabase::getPendingAlerts()
        {
            std::vector<DBAlertEvent> pendingAlerts;

            if (!isInitialized_)
            {
                return pendingAlerts;
            }

            std::string querySQL =
                "SELECT "
                "id, "
                "event_id, "
                "device_id, "
                "deployment_area, "
                "event_type, "
                "sources, "
                "timestamp, "
                "person_track_id, "
                "trigger_x, "
                "trigger_y, "
                "keyword, "
                "video_path, "
                "event_status "
                "FROM alerts "
                "WHERE upload_status = 'PENDING' "
                "ORDER BY id ASC;";

            sqlite3_stmt* stmt = nullptr;

            if (sqlite3_prepare_v2(
                    db_,
                    querySQL.c_str(),
                    -1,
                    &stmt,
                    nullptr) != SQLITE_OK)
            {
                LOG_ERROR(
                    "查询待补传报警失败：{}",
                    sqlite3_errmsg(db_)
                );

                return pendingAlerts;
            }

            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                DBAlertEvent alert;

                alert.dbId =
                    sqlite3_column_int(stmt, 0);

                auto readText =
                    [&](int column) -> std::string
                {
                    const unsigned char* value =
                        sqlite3_column_text(
                            stmt,
                            column
                        );

                    return value
                        ? reinterpret_cast<const char*>(value)
                        : "";
                };

                alert.eventId =
                    readText(1);

                alert.deviceId =
                    readText(2);

                alert.deploymentArea =
                    readText(3);

                std::string eventType =
                    readText(4);

                if (eventType == "FALL")
                {
                    alert.eventType =
                        event::EventType::FALL;
                }
                else
                {
                    alert.eventType =
                        event::EventType::HELP_REQUEST;
                }

                // -----------------------------
                // 解析报警来源
                // -----------------------------
                alert.source.clear();

                std::string sources =
                    readText(5);

                std::stringstream ss(sources);
                std::string source;

                while (std::getline(ss,source,','))
                {
                    if (source == "VISION")
                    {
                        alert.source.push_back(
                            event::EventSource::VISION
                        );
                    }
                    else if (source == "VOICE")
                    {
                        alert.source.push_back(
                            event::EventSource::VOICE
                        );
                    }
                }

                alert.timestamp =
                    sqlite3_column_int64(stmt, 6);

                alert.personTrackId =
                    sqlite3_column_int(stmt, 7);

                alert.triggerBoxX =
                    sqlite3_column_double(stmt, 8);

                alert.triggerBoxY =
                    sqlite3_column_double(stmt, 9);

                alert.keyword =
                    readText(10);

                alert.videoPath =
                    readText(11);

                std::string eventStatus =
                    readText(12);

                if (eventStatus == "ACKNOWLEDGED")
                {
                    alert.status =
                        event::EventStatus::ACKNOWLEDGED;
                }
                else if (eventStatus == "RESOLVED")
                {
                    alert.status =
                        event::EventStatus::RESOLVED;
                }
                else
                {
                    alert.status =
                        event::EventStatus::NEW;
                }

                pendingAlerts.push_back(
                    std::move(alert)
                );
            }

            sqlite3_finalize(stmt);

            return pendingAlerts;
        }

        bool LocalDatabase::markAsUploaded(const std::string& eventId)
        {
            if (!isInitialized_)
            {
                return false;
            }

            std::string safeEventId;

            for (char ch : eventId)
            {
                if (ch == '\'')
                {
                    safeEventId += "''";
                }
                else
                {
                    safeEventId += ch;
                }
            }

            std::string updateSQL =
                "UPDATE alerts "
                "SET upload_status = 'UPLOADED' "
                "WHERE event_id = '" +
                safeEventId +
                "';";

            {
                std::lock_guard<std::mutex> lock(queueMtx_);
                sqlQueue_.push(std::move(updateSQL));
            }

            cv_.notify_one();

            LOG_INFO(
                "报警事件已标记为已上传：event_id={}",
                eventId
            );

            return true;
        }

        void LocalDatabase::dbWorkerLoop()
        {
            while (isRunning_)
            {
                std::string sql;
                
                // 1. 阻塞等待 SQL 任务，零 CPU 消耗
                {
                    std::unique_lock<std::mutex> lock(queueMtx_);
                    cv_.wait(lock, [this]() { return !sqlQueue_.empty() || !isRunning_; });

                    if (!isRunning_ && sqlQueue_.empty())
                    {
                        break;
                    }

                    sql = std::move(sqlQueue_.front());
                    sqlQueue_.pop();
                }

                // 2. 在锁外执行极其耗时的底层磁盘 I/O 写入
                executeSQL(sql);
            }
        }
    }
}