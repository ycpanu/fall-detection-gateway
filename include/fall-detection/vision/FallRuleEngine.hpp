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
         * @brief 摔倒规则引擎类
         * 负责将 NPU 输出的单帧静态特征通过时序追踪和几何计算，转化为动态的摔倒过程判断
         */
        class FallRuleEngine
        {
            public:
                FallRuleEngine();
                ~FallRuleEngine() = default;

                /**
                 * @brief 处理单帧 AI 推理结果
                 * @param aiResults NPU 这一帧识别出的人体边界框列表
                 * @param outEvent 如果判定摔倒，将报警信息写入该结构体
                 * @return true 代表认为异常摔倒，false 代表正常或过滤
                 */
                bool processFrame(const std::vector<DetectResult>& aiResults, AlertEvent& outEvent);

            private:
                // 内部状态追踪器，用于计算时序和速度
                bool hasPreviousTarget_;                                // 上一帧是否检测到有效人体
                int previousCenterY_;                                   // 上一帧人体中心点的 Y 坐标
                int previousClassId_;                                   // 上一帧人体姿态类别    
                std::chrono::steady_clock::time_point lastTime_;        // 上一帧时间戳

                int lieConfirmCount_;                                   // 连续处于 lie 状态的帧技术器

                // 阈值配置，可根据摄像头安装高度进行调优
                const int CONFIRM_FRAMES_THRESHOLD = 5;         // 需连续 5 帧 (约 0.3 秒) 判定为 lie 才报警
                const float FALL_VELOCITY_THRESHOLD = 500.0f;   // Y 轴下坠速度阈值 (像素/秒)
                const int CLASS_LIE = 3;                        // 假设 3 代表 lie (模型调整)
        };
    };
}