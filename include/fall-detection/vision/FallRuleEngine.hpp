#pragma once
#include <unordered_map>
#include <vector>
#include <chrono>
#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/vision/SafeZoneManager.hpp"
#include "fall-detection/vision/PersonTracker.hpp"
#include "fall-detection/event/AlertEvent.hpp"

namespace fall_detection
{
    namespace vision
    {
        using AlertEvent = event::AlertEvent;
        /**
         * @brief 摔倒规则引擎参数配置（由 ConfigManager 从 config.json 读取后注入）
         */
        struct FallRuleConfig
        {
            // 关键点置信阈值
            float kptConfThreshold = 0.3f;

            // 躯干夹角阈值
            float fallAngleThreshold = 60.0f;

            float recoveryAngleThreshold = 35.0f;

            // 归一化髋部下坠速度阈值
            float normalizedVelocityThreshold = 0.6f;

            // 人体中心归一化下降速度阈值
            float normalizedCenterVelocityThreshold = 0.5f;

            // 疑似跌倒后，保持躺倒多长时间才确认报警
            int suspectConfirmMs = 1000;

            // 没捕获快速下坠时，持续异常躺卧多久进行静态兜底报警
            int staticLieConfirmMs = 5000;

            // 快速下坠事件在多长时间内仍然有效
            int fallEventWindowMs = 1500;

            // 人体框与安全区域最小重叠比例
            float safeZoneOverlapThreshold = 0.5f;
        };

        /**
         * @brief 跌倒状态
         */
        enum class FallState
        {
            NORMAL,             // 正常状态
            SUSPECTED_FALL,     // 疑似跌倒
            CONFIRMED_FALL,     // 已确认跌倒
            ALARMED             // 已产生报警，等待人体恢复
        };

        /**
         * @brief 跌倒规则引擎
         *
         * 核心依据：
         * 1. 躯干倾角
         * 2. 归一化髋部下降速度
         * 3. 人体整体中心下降
         * 4. 基于实际时间的状态机
         */
        class FallRuleEngine
        {
            public:
                explicit FallRuleEngine(const FallRuleConfig& config = FallRuleConfig(), const SafeZoneManager* safeZoneManager = nullptr);
                ~FallRuleEngine() = default;

                /**
                 * @brief 处理单帧 AI 推理结果
                 * @param aiResults NPU 这一帧识别出的人体（含 17 骨骼关键点）列表
                 * @param outEvent 如果判定摔倒，将报警信息写入该结构体
                 * @return true 代表认为异常摔倒，false 代表正常或过滤
                 */
                std::vector<AlertEvent> processFrame(const std::vector<TrackedPerson>& trackedPersons, int frameWidth, int frameHeight);

            private:
                struct PersonState
                {
                    // 当前跌倒状态
                    FallState state = FallState::NORMAL;

                    // 是否已经有上一帧数据
                    bool hasPreviousTarget = false;

                    // 上一帧运动数据
                    float previousHipY = 0.0f;
                    float previousCenterY = 0.0f;
                    float previousBodyHeight = 0.0f;

                    std::chrono::steady_clock::time_point lastFrameTime;

                    // 状态机时间
                    std::chrono::steady_clock::time_point suspectedStartTime;
                    std::chrono::steady_clock::time_point lieStartTime;
                    std::chrono::steady_clock::time_point lastFastDropTime;

                    // 最近一次真正检测到这个人的时间
                    std::chrono::steady_clock::time_point lastSeenTime;

                    bool lieTimerActive = false;
                    bool fastDropDetected = false;
                };

                std::unordered_map<int, PersonState> personStates_;
                // 配置
                FallRuleConfig config_;

                // 安全区域管理器
                const SafeZoneManager* safeZoneManager_ = nullptr;

            private:
                bool processPerson(
                    const TrackedPerson& person,
                    int frameWidth,
                    int frameHeight,
                    PersonState& personState,
                    AlertEvent& outEvent);

                void resetPersonToNormal(
                    PersonState& personState);

                void removeExpiredPersonStates(
                    const std::chrono::steady_clock::time_point& now);
        };
    }
}
