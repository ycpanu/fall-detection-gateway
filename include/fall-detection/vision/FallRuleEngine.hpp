#pragma once

#include <vector>
#include <chrono>
#include "fall-detection/vision/RKNNInferencer.hpp"

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
            float kptConfThreshold = 0.3f;        // 关键点置信度阈值（低于视为不可见）
            float fallAngleThreshold = 60.0f;     // 身体轴线(肩→髋)与垂直方向夹角阈值（度）
            float fallVelocityThreshold = 400.0f; // 髋部下坠速度阈值（像素/秒）
            int confirmFramesThreshold = 5;       // 下坠后需连续躺倒的报警帧数
            int staticLieThreshold = 30;          // 无下坠时持续躺倒的兜底报警帧数
            int fallEventWindow = 15;             // 快速下坠事件的有效窗口（帧）
        };

        /**
         * @brief 摔倒规则引擎类
         * 通过身体轴线角度 + 髋部下坠速度 + 时序状态机，将静态关键点转化为动态摔倒判定
         */
        class FallRuleEngine
        {
            public:
                explicit FallRuleEngine(const FallRuleConfig& config = FallRuleConfig());
                ~FallRuleEngine() = default;

                /**
                 * @brief 处理单帧 AI 推理结果
                 * @param aiResults NPU 这一帧识别出的人体（含 17 骨骼关键点）列表
                 * @param outEvent 如果判定摔倒，将报警信息写入该结构体
                 * @return true 代表认为异常摔倒，false 代表正常或过滤
                 */
                bool processFrame(const std::vector<DetectResult>& aiResults, AlertEvent& outEvent);

            private:
                void resetState();

                FallRuleConfig config_;                     // 运行时阈值配置
                int lieConfirmCount_;                       // 躺倒连续帧计数（防抖）
                bool hasPreviousTarget_;                    // 上一帧是否有有效目标（用于算速度）
                float previousHipY_;                        // 上一帧髋部中点 Y 坐标
                std::chrono::steady_clock::time_point lastTime_;  // 上一帧时间戳
                bool fallEventPending_;                     // 快速下坠事件是否仍在时序窗口内
                int fallEventFrames_;                       // 下坠事件发生后经过的帧数
        };
    }
}
