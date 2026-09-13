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
                "timestamp INTEGER NOT NULL, "
                "trigger_x INTEGER NOT NULL, "
                "trigger_y INTEGER NOT NULL, "
                "status TEXT NOT NULL);";

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

        bool LocalDatabase::saveAlert(const vision::AlertEvent& event)
        {
            if (!isInitialized_) return false;

            std::string insertSQL = "INSERT INTO alerts (timestamp, trigger_x, trigger_y, status) VALUES (" +
                                std::to_string(event.timestamp) + ", " +
                                std::to_string(event.triggerBoxX) + ", " +
                                std::to_string(event.triggerBoxY) + ", 'pending');";
            
            // 极速内存入队，绝不在此处进行磁盘 I/O 阻塞
            {
                std::lock_guard<std::mutex> lock(queueMtx_);
                sqlQueue_.push(std::move(insertSQL));
            }
            cv_.notify_one();
            
            LOG_WARN("已触发断网容灾存储：报警数据进入异步写盘队列，等待网络恢复。");
            return true;
        }
        
        std::vector<DBAlertEvent> LocalDatabase::getPendingAlerts()
        {
            std::vector<DBAlertEvent> pendingAlerts;
            if (!isInitialized_) return pendingAlerts;

            std::string querySQL = "SELECT id, timestamp, trigger_x, trigger_y FROM alerts WHERE status = 'pending';";
            sqlite3_stmt* stmt;

            if (sqlite3_prepare_v2(db_, querySQL.c_str(), -1, &stmt, nullptr) == SQLITE_OK)
            {
                while (sqlite3_step(stmt) == SQLITE_ROW)
                {
                    DBAlertEvent event;
                    event.dbId = sqlite3_column_int(stmt, 0);
                    event.timestamp = sqlite3_column_int64(stmt, 1);
                    event.triggerBoxX = sqlite3_column_int(stmt, 2);
                    event.triggerBoxY = sqlite3_column_int(stmt, 3);
                    event.isFall = true;
                    pendingAlerts.push_back(event);
                }
                sqlite3_finalize(stmt);
            }
            else
            {
                LOG_ERROR("查询待上传报警记录失败：{}", sqlite3_errmsg(db_));
            }

            return pendingAlerts;
        }

        bool LocalDatabase::markAsUploaded(int id)
        {
            if (!isInitialized_) return false;
            std::string updateSQL = "UPDATE alerts SET status = 'uploaded' WHERE id = " + std::to_string(id) + ";";
            
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