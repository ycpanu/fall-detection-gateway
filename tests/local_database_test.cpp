#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <sqlite3.h>
#include <unistd.h>

#include "fall-detection/utils/LocalDatabase.hpp"
#include "fall-detection/utils/SysLogger.hpp"

using namespace fall_detection;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "检查失败：" << #condition << " (" << __LINE__ << ")\n"; \
        std::abort(); \
    } \
} while (false)

namespace
{
    std::string queryText(sqlite3* connection, const char* sql)
    {
        sqlite3_stmt* statement = nullptr;
        CHECK(sqlite3_prepare_v2(connection, sql, -1, &statement, nullptr) == SQLITE_OK);
        CHECK(sqlite3_step(statement) == SQLITE_ROW);
        const auto* value = sqlite3_column_text(statement, 0);
        const std::string result = value ? reinterpret_cast<const char*>(value) : "";
        sqlite3_finalize(statement);
        return result;
    }
}

int main()
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("fall_database_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    const auto dbPath = (directory / "alerts.db").string();
    utils::SysLogger::getInstance().init((directory / "test.log").string());

    {
        utils::LocalDatabase db(dbPath);
        CHECK(db.init());
        event::AlertEvent event;
        event.eventId = "event-1";
        event.deviceId = "test-device";
        event.eventType = event::EventType::FALL;
        event.source = {event::EventSource::VISION};
        event.timestamp = 123;
        CHECK(db.saveAlert(event));

        // 从另一连接读取，确认 saveAlert 返回时事务已提交。
        sqlite3* reader = nullptr;
        CHECK(sqlite3_open(dbPath.c_str(), &reader) == SQLITE_OK);
        CHECK(queryText(reader,
            "SELECT upload_status FROM alerts WHERE event_id='event-1';") == "PENDING");

        CHECK(db.markAsUploaded(event.eventId));
        CHECK(queryText(reader,
            "SELECT upload_status FROM alerts WHERE event_id='event-1';") == "UPLOADED");

        event.keyword = "更新";
        CHECK(db.saveAlert(event));
        CHECK(queryText(reader,
            "SELECT upload_status FROM alerts WHERE event_id='event-1';") == "PENDING");

        CHECK(db.markVideoPending(event.eventId, "/tmp/test.mp4"));
        CHECK(queryText(reader,
            "SELECT video_upload_status FROM alerts WHERE event_id='event-1';") == "PENDING");
        CHECK(db.markVideoUploaded(event.eventId));
        CHECK(queryText(reader,
            "SELECT video_upload_status FROM alerts WHERE event_id='event-1';") == "UPLOADED");

        // SQLite 拒绝写入时不能把报警当作已经持久化。
        CHECK(sqlite3_exec(reader,
            "CREATE TRIGGER reject_alert BEFORE INSERT ON alerts "
            "WHEN NEW.event_id='reject' BEGIN SELECT RAISE(ABORT, 'rejected'); END;",
            nullptr, nullptr, nullptr) == SQLITE_OK);
        event.eventId = "reject";
        CHECK(!db.saveAlert(event));
        sqlite3_close(reader);
    }

    std::filesystem::remove_all(directory);
    return 0;
}
