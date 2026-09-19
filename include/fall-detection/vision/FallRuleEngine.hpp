#pragma once

#include <vector>
#include <chrono>
#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/vision/SafeZoneManager.hpp"

namespace fall_detection
{
    namespace vision
    {
        /**
         * @brief 摔倒事件预警结构体
         * 用于记录出发报警时的关键信息，准备打包为 JSON 发送给 MQTT
         */
        struct AlertEvent
        {
            bool isFall;            // 是否确认发生摔倒
            long long timestamp;    // 发生时间戳
            int triggerBoxX;        // 触发报警时的目标中心点 X
            int triggerBoxY;        // 目标中心点 Y
            std::string videoPath;  // 摔倒现场视频路径
        };

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
                bool processFrame(const std::vector<DetectResult>& aiResults, int frameWidth, int frameHeight, AlertEvent& outEvent);

            private:
                /**
                 * @brief 完整清空规则引擎状态
                 *
                 * 用于长时间没有有效目标等情况。
                 */
                void resetState();

                /**
                 * @brief 恢复到 NORMAL 状态
                 *
                 * 与 resetState() 区分开，
                 * 后续状态机实现时更清晰。
                 */
                void resetToNormal();

           private:
                // 配置
                FallRuleConfig config_;

                // 当前状态机状态
                FallState state_ = FallState::NORMAL;

                // 上一帧人体运动信息
                bool hasPreviousTarget_ = false;

                // 上一帧髋部中点 Y
                float previousHipY_ = 0.0f;

                // 上一帧人体检测框中心 Y
                float previousCenterY_ = 0.0f;

                // 上一帧人体检测框高度
                // 用于运动特征归一化
                float previousBodyHeight_ = 0.0f;


                // 上一帧处理时间
                std::chrono::steady_clock::time_point lastFrameTime_;

                // 进入 SUSPECTED_FALL 的时间
                std::chrono::steady_clock::time_point suspectedStartTime_;

                // 开始持续躺倒的时间
                std::chrono::steady_clock::time_point lieStartTime_;

                // 最近一次检测到明显快速下降的时间
                std::chrono::steady_clock::time_point lastFastDropTime_;

                // 当前是否正在统计静态躺倒时间
                bool lieTimerActive_ = false;

                // 最近是否发生过快速下降
                bool fastDropDetected_ = false;

                // 安全区域管理器
                const SafeZoneManager* safeZoneManager_ = nullptr;
        };
    }
}
