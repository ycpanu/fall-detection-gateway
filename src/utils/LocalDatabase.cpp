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


            // sources 转成字符串，例如：
            // VISION
            // VOICE
            // VISION,VOICE

            std::string sourcesStr;

            for (std::size_t i = 0;
                i < alertEvent.source.size();
                ++i)
            {
                if (i > 0)
                {
                    sourcesStr += ",";
                }

                sourcesStr +=
                    event::toString(
                        alertEvent.source[i]
                    );
            }


            std::string insertSQL =
                "INSERT OR IGNORE INTO alerts ("
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
                ") VALUES ('"

                + alertEvent.eventId + "', '"
                + alertEvent.deviceId + "', '"
                + alertEvent.deploymentArea + "', '"
                + event::toString(alertEvent.eventType) + "', '"
                + sourcesStr + "', "
                + std::to_string(alertEvent.timestamp) + ", "
                + std::to_string(alertEvent.personTrackId) + ", "
                + std::to_string(alertEvent.triggerBoxX) + ", "
                + std::to_string(alertEvent.triggerBoxY) + ", '"
                + alertEvent.keyword + "', '"
                + alertEvent.videoPath + "', '"
                + event::toString(alertEvent.status) + "', "
                "'pending');";

            {
                std::lock_guard<std::mutex> lock(
                    queueMtx_
                );

                sqlQueue_.push(
                    std::move(insertSQL)
                );
            }


            cv_.notify_one();


            LOG_WARN(
                "报警事件已进入本地持久化队列："
                "event_id={}, type={}",
                alertEvent.eventId,
                event::toString(
                    alertEvent.eventType
                )
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

            const std::string querySQL =
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
                "WHERE upload_status = 'pending';";

            sqlite3_stmt* stmt = nullptr;

            if (sqlite3_prepare_v2(
                    db_,
                    querySQL.c_str(),
                    -1,
                    &stmt,
                    nullptr) != SQLITE_OK)
            {
                LOG_ERROR(
                    "查询待上传报警记录失败：{}",
                    sqlite3_errmsg(db_)
                );

                return pendingAlerts;
            }

            while (sqlite3_step(stmt) ==
                SQLITE_ROW)
            {
                DBAlertEvent alertEvent;


                alertEvent.dbId =
                    sqlite3_column_int(
                        stmt,
                        0
                    );

                const auto* eventId =
                    sqlite3_column_text(
                        stmt,
                        1
                    );

                alertEvent.eventId =
                    eventId
                        ? reinterpret_cast<
                            const char*>(eventId)
                        : "";

                const auto* deviceId =
                    sqlite3_column_text(
                        stmt,
                        2
                    );

                const auto* deploymentArea =
                    sqlite3_column_text(
                        stmt,
                        3
                    );

                alertEvent.deploymentArea =
                    deploymentArea
                        ? reinterpret_cast<
                            const char*>(deploymentArea)
                        : "";

                alertEvent.deviceId =
                    deviceId
                        ? reinterpret_cast<
                            const char*>(deviceId)
                        : "";


                // event_type
                std::string eventTypeStr;

                const auto* eventType =
                    sqlite3_column_text(
                        stmt,
                        4
                    );

                if (eventType)
                {
                    eventTypeStr =
                        reinterpret_cast<
                            const char*>(eventType);
                }


                if (eventTypeStr == "HELP_REQUEST")
                {
                    alertEvent.eventType =
                        event::EventType::HELP_REQUEST;
                }
                else
                {
                    alertEvent.eventType =
                        event::EventType::FALL;
                }


                // sources
                std::string sourcesStr;

                const auto* sources =
                    sqlite3_column_text(
                        stmt,
                        5
                    );

                if (sources)
                {
                    sourcesStr =
                        reinterpret_cast<
                            const char*>(sources);
                }


                if (sourcesStr.find("VISION") !=
                    std::string::npos)
                {
                    alertEvent.source.push_back(
                        event::EventSource::VISION
                    );
                }

                if (sourcesStr.find("VOICE") !=
                    std::string::npos)
                {
                    alertEvent.source.push_back(
                        event::EventSource::VOICE
                    );
                }


                alertEvent.timestamp =
                    sqlite3_column_int64(
                        stmt,
                        6
                    );

                alertEvent.personTrackId =
                    sqlite3_column_int(
                        stmt,
                        7
                    );

                alertEvent.triggerBoxX =
                    sqlite3_column_int(
                        stmt,
                        8
                    );

                alertEvent.triggerBoxY =
                    sqlite3_column_int(
                        stmt,
                        9
                    );


                const auto* keyword =
                    sqlite3_column_text(
                        stmt,
                        10
                    );

                alertEvent.keyword =
                    keyword
                        ? reinterpret_cast<
                            const char*>(keyword)
                        : "";


                const auto* videoPath =
                    sqlite3_column_text(
                        stmt,
                        11
                    );

                alertEvent.videoPath =
                    videoPath
                        ? reinterpret_cast<
                            const char*>(videoPath)
                        : "";


                // 当前 pending 事件读取回来时，
                // 业务状态默认为 NEW。
                // 后续后台 ACK/RESOLVED 再完善。
                alertEvent.status =
                    event::EventStatus::NEW;


                pendingAlerts.push_back(
                    std::move(alertEvent)
                );
            }


            sqlite3_finalize(stmt);

            return pendingAlerts;
        }

        bool LocalDatabase::markAsUploaded(int id)
        {
            if (!isInitialized_) return false;
            std::string updateSQL =
                "UPDATE alerts "
                "SET upload_status = 'uploaded' "
                "WHERE id = "
                + std::to_string(id)
                + ";";
                
            // 同样放入异步队列更新状态，消除同步等待
            {
                std::lock_guard<std::mutex> lock(queueMtx_);
                sqlQueue_.push(std::move(updateSQL));
            }
            cv_.notify_one();
            
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