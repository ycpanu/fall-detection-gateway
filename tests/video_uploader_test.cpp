#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <unistd.h>

#include "fall-detection/network/VideoUploader.hpp"
#include "fall-detection/utils/SysLogger.hpp"

using namespace fall_detection;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "检查失败：" << #condition << " (" << __LINE__ << ")\n"; \
        std::abort(); \
    } \
} while (false)

int main()
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("fall_uploader_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    const auto marker = directory / "upload_called";
    const auto fakeCurl = directory / "curl";
    const auto video = directory / "event.mp4";
    {
        std::ofstream script(fakeCurl);
        script << "#!/bin/sh\nprintf x >> " << marker.string() << "\n";
    }
    std::filesystem::permissions(fakeCurl,
        std::filesystem::perms::owner_exec | std::filesystem::perms::owner_read |
        std::filesystem::perms::owner_write, std::filesystem::perm_options::replace);
    std::ofstream(video).put('x');
    const char* oldPath = std::getenv("PATH");
    const std::string path = directory.string() + ":" +
        (oldPath ? oldPath : "/usr/bin:/bin");
    CHECK(::setenv("PATH", path.c_str(), 1) == 0);
    utils::SysLogger::getInstance().init((directory / "test.log").string());

    {
        utils::LocalDatabase database((directory / "alerts.db").string());
        CHECK(database.init());
        event::AlertEvent alert;
        alert.eventId = "upload-1";
        alert.deviceId = "test-device";
        alert.source = {event::EventSource::VISION};
        alert.timestamp = 123;
        CHECK(database.saveAlert(alert));
        CHECK(database.markVideoPending(alert.eventId, video.string()));

        network::VideoUploader uploader(database, "http://127.0.0.1:1", 5);
        uploader.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        CHECK(!std::filesystem::exists(marker));

        uploader.setOnline(true);
        for (int i = 0; i < 100 && !std::filesystem::exists(marker); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK(std::filesystem::exists(marker));
        uploader.stop();
        CHECK(database.getPendingVideos().empty());
    }

    std::filesystem::remove_all(directory);
    return 0;
}
