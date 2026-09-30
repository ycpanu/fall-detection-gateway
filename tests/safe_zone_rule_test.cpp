#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/vision/SafeZoneManager.hpp"

using namespace fall_detection;
using namespace fall_detection::vision;
using namespace std::chrono_literals;

namespace
{
constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;

TrackedPerson makePerson(
    int trackId,
    int boxX,
    int boxY,
    int boxWidth,
    int boxHeight,
    int shoulderX,
    int shoulderY,
    int hipX,
    int hipY)
{
    TrackedPerson person;
    person.trackId = trackId;

    auto& detection = person.detection;
    detection.classId = 0;
    detection.confidence = 0.95f;
    detection.x = boxX;
    detection.y = boxY;
    detection.width = boxWidth;
    detection.height = boxHeight;

    detection.keypoints.resize(17);

    for (auto& point : detection.keypoints)
    {
        point.x = 0;
        point.y = 0;
        point.confidence = 0.0f;
    }

    // COCO: 5/6 = 左右肩，11/12 = 左右髋
    detection.keypoints[5] = {shoulderX - 10, shoulderY, 0.95f};
    detection.keypoints[6] = {shoulderX + 10, shoulderY, 0.95f};

    detection.keypoints[11] = {hipX - 10, hipY, 0.95f};
    detection.keypoints[12] = {hipX + 10, hipY, 0.95f};

    return person;
}

bool hasFallEvent(const std::vector<event::AlertEvent>& events)
{
    return !events.empty();
}

bool createTestSceneConfig(const std::string& path)
{
    std::ofstream file(path);

    if (!file.is_open())
        return false;

    file << R"({
  "version": 1,
  "safe_lie_zones": [
    {
      "name": "test_bed",
      "points": [
        [0.10, 0.40],
        [0.90, 0.40],
        [0.90, 0.90],
        [0.10, 0.90]
      ]
    }
  ]
})";

    return file.good();
}

FallRuleConfig createTestRuleConfig()
{
    FallRuleConfig config;

    config.kptConfThreshold = 0.3f;
    config.fallAngleThreshold = 60.0f;
    config.recoveryAngleThreshold = 35.0f;

    config.normalizedVelocityThreshold = 0.6f;
    config.normalizedCenterVelocityThreshold = 0.5f;

    // 测试时缩短时间，不需要真的等待 5 秒。
    config.suspectConfirmMs = 60;
    config.staticLieConfirmMs = 90;
    config.fallEventWindowMs = 300;

    config.safeZoneOverlapThreshold = 0.5f;

    return config;
}

bool testStaticLyingInsideSafeZone(SafeZoneManager& safeZoneManager)
{
    std::cout << "\n[TEST 1] 安全区域内静态躺卧" << std::endl;

    FallRuleEngine engine(createTestRuleConfig(), &safeZoneManager);

    // 身体轴线接近水平：
    // shoulder=(220,300)
    // hip=(320,300)
    // angle ≈ 90°
    //
    // 人体框完全位于测试安全区域内。
    const TrackedPerson person = makePerson(
        1,
        160,
        220,
        320,
        160,
        220,
        300,
        320,
        300
    );

    bool alarmed = false;

    for (int i = 0; i < 7; ++i)
    {
        const auto events = engine.processFrame(
            {person},
            FRAME_WIDTH,
            FRAME_HEIGHT
        );

        if (hasFallEvent(events))
            alarmed = true;

        std::this_thread::sleep_for(25ms);
    }

    if (alarmed)
    {
        std::cout << "[FAIL] 安全区域内静态躺卧产生了 FALL" << std::endl;
        return false;
    }

    std::cout << "[PASS] 安全区域内静态躺卧未产生 FALL" << std::endl;
    return true;
}

bool testStaticLyingOutsideSafeZone(SafeZoneManager& safeZoneManager)
{
    std::cout << "\n[TEST 2] 安全区域外静态躺卧" << std::endl;

    FallRuleEngine engine(createTestRuleConfig(), &safeZoneManager);

    // 同样是水平躺卧，但人体位于安全区域上方。
    const TrackedPerson person = makePerson(
        2,
        160,
        30,
        320,
        120,
        220,
        100,
        320,
        100
    );

    bool alarmed = false;

    for (int i = 0; i < 7; ++i)
    {
        const auto events = engine.processFrame(
            {person},
            FRAME_WIDTH,
            FRAME_HEIGHT
        );

        if (hasFallEvent(events))
        {
            alarmed = true;
            break;
        }

        std::this_thread::sleep_for(25ms);
    }

    if (!alarmed)
    {
        std::cout << "[FAIL] 安全区域外持续躺卧没有产生 FALL" << std::endl;
        return false;
    }

    std::cout << "[PASS] 安全区域外持续躺卧正常产生 FALL" << std::endl;
    return true;
}

bool testFastDropIntoSafeZone(SafeZoneManager& safeZoneManager)
{
    std::cout << "\n[TEST 3] 快速下降后进入安全区域" << std::endl;

    FallRuleEngine engine(createTestRuleConfig(), &safeZoneManager);

    const TrackedPerson standing = makePerson(
        3,
        260,
        170,
        80,
        220,
        300,
        220,
        300,
        320
    );

    // 先输入站立帧，建立上一帧运动状态。
    engine.processFrame(
        {standing},
        FRAME_WIDTH,
        FRAME_HEIGHT
    );

    std::this_thread::sleep_for(25ms);

    const TrackedPerson lying = makePerson(
        3,
        160,
        220,
        320,
        160,
        220,
        300,
        320,
        300
    );

    bool alarmed = false;

    for (int i = 0; i < 6; ++i)
    {
        const auto events = engine.processFrame(
            {lying},
            FRAME_WIDTH,
            FRAME_HEIGHT
        );

        if (hasFallEvent(events))
        {
            alarmed = true;
            break;
        }

        std::this_thread::sleep_for(30ms);
    }

    if (alarmed)
    {
        std::cout << "[FAIL] 快速下降后落入安全区域仍产生了 FALL" << std::endl;
        return false;
    }

    std::cout << "[PASS] 快速下降后落入安全区域被正确抑制" << std::endl;
    return true;
}

bool testFastDropOutsideSafeZone(SafeZoneManager& safeZoneManager)
{
    std::cout << "\n[TEST 4] 安全区域外快速下降" << std::endl;

    FallRuleEngine engine(createTestRuleConfig(), &safeZoneManager);

    const TrackedPerson standing = makePerson(
        4,
        260,
        0,
        80,
        100,
        300,
        20,
        300,
        80
    );

    engine.processFrame(
        {standing},
        FRAME_WIDTH,
        FRAME_HEIGHT
    );

    std::this_thread::sleep_for(25ms);

    // 中心点和髋部明显向下移动，同时身体变成水平状态。
    const TrackedPerson lying = makePerson(
        4,
        160,
        70,
        320,
        100,
        220,
        120,
        320,
        120
    );

    bool alarmed = false;

    for (int i = 0; i < 6; ++i)
    {
        const auto events = engine.processFrame(
            {lying},
            FRAME_WIDTH,
            FRAME_HEIGHT
        );

        if (hasFallEvent(events))
        {
            alarmed = true;
            break;
        }

        std::this_thread::sleep_for(30ms);
    }

    if (!alarmed)
    {
        std::cout << "[FAIL] 安全区域外快速下降没有产生 FALL" << std::endl;
        return false;
    }

    std::cout << "[PASS] 安全区域外快速下降正常产生 FALL" << std::endl;
    return true;
}
}

int main()
{
    utils::SysLogger::getInstance().init("/tmp/safe_zone_rule_test.log");

    const std::string configPath = "/tmp/safe_zone_rule_test.json";

    if (!createTestSceneConfig(configPath))
    {
        std::cerr << "[ERROR] 无法创建测试场景配置" << std::endl;
        return 1;
    }

    SafeZoneManager safeZoneManager;

    if (!safeZoneManager.load(configPath))
    {
        std::cerr << "[ERROR] SafeZoneManager 加载测试配置失败" << std::endl;
        std::remove(configPath.c_str());
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << " SafeZone / FallRuleEngine 自动化测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "安全区域数量：" << safeZoneManager.getZoneCount() << std::endl;

    int passed = 0;
    int total = 0;

    ++total;
    if (testStaticLyingInsideSafeZone(safeZoneManager))
        ++passed;

    ++total;
    if (testStaticLyingOutsideSafeZone(safeZoneManager))
        ++passed;

    ++total;
    if (testFastDropIntoSafeZone(safeZoneManager))
        ++passed;

    ++total;
    if (testFastDropOutsideSafeZone(safeZoneManager))
        ++passed;

    std::remove(configPath.c_str());

    std::cout << "\n========================================" << std::endl;
    std::cout << "测试结果：" << passed << " / " << total << " 通过" << std::endl;
    std::cout << "========================================" << std::endl;

    return passed == total ? 0 : 1;
}