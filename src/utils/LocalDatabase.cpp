#include "fall-detection/utils/LocalDatabase.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace utils
    {
        LocalDatabase::LocalDatabase(const std::string& dbPath) : dbPath_(dbPath), db_(nullptr), isInitialized_(false){}
        LocalDatabase::~LocalDatabase()
        {
            if (db_ != nullptr)
            {
                sqlite3_close(db_);
                LOG_INFO("本地数据库连接已安全关闭！");
            }
        }

        bool LocalDatabase::init()
        {
            // 打开数据库，如果文件不存在则自动创建
            if (sqlite3_open(dbPath_.c_str(), &db_) != SQLITE_OK)
            {
                LOG_ERROR("无法打开本地数据库：{}", sqlite3_errmsg(db_));
                sqlite3_close(db_);
                db_ = nullptr;
                return false;
            }

            // 创建报警记录表
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
            LOG_INFO("本地断网缓存数据库初始化成功！文件路径：{}", dbPath_);
            return true;
        }

        bool LocalDatabase::executeSQL(const std::string& sql)
        {
            char* errMsg = nullptr;

            // 执行 SQL 语句
            if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK)
            {
                LOG_ERROR("SQL 执行错误：{}", errMsg);
                sqlite3_free(errMsg);
                return false;
            }
            return true;
        }

        bool LocalDatabase::saveAlert(const vision::AlertEvent& event)
        {
            if (!isInitialized_) return false;

            // 拼接 INSERT 语句，默认 status 为 'pending'
            std::string insertSQL = "INSERT INTO alerts (timestamp, trigger_x, trigger_y, status) VALUES (" +
                                std::to_string(event.timestamp) + ", " +
                                std::to_string(event.triggerBoxX) + ", " +
                                std::to_string(event.triggerBoxY) + ", 'pending');";
            
            if (executeSQL(insertSQL))
            {
                LOG_WARN("已触发断网容灾存储：报警数据落盘至 SQLite，等待网络恢复。");
                return true;
            }
            return false;
        }
        
        std::vector<DBAlertEvent> LocalDatabase::getPendingAlerts()
        {
            std::vector<DBAlertEvent> pendingAlerts;
            if (!isInitialized_) return pendingAlerts;

            std::string querySQL = "SELECT id, timestamp, trigger_x, trigger_y FROM alerts WHERE status = 'pending';";
            sqlite3_stmt* stmt;

            // 编译 SQL 查询语句
            if (sqlite3_prepare_v2(db_, querySQL.c_str(), -1, &stmt, nullptr) == SQLITE_OK)
            {
                // 逐行提取查询结果
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
                sqlite3_finalize(stmt);     // 释放游标
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
            return executeSQL(updateSQL);
        }
    }
}