#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <cmath>
#include <algorithm>

namespace fall_detection
{
    namespace vision
    {
        FallRuleEngine::FallRuleEngine()
            : lieConfirmCount_(0), hasPreviousTarget_(false), previousHipY_(0.0f),
              fallEventPending_(false), fallEventFrames_(0)
        {
            lastTime_ = std::chrono::steady_clock::now();
        }

        void FallRuleEngine::resetState()
        {
            hasPreviousTarget_ = false;
            lieConfirmCount_ = 0;
            fallEventPending_ = false;
            fallEventFrames_ = 0;
        }

        bool FallRuleEngine::processFrame(const std::vector<DetectResult>& aiResults, AlertEvent& outEvent)
        {
            outEvent.isFall = false;

            // COCO 17 关键点索引（YOLOv8-Pose 标准）
            constexpr int L_SHOULDER = 5;
            constexpr int R_SHOULDER = 6;
            constexpr int L_HIP = 11;
            constexpr int R_HIP = 12;

            // 1. 当前帧没检测到人 → 重置
            if (aiResults.empty())
            {
                resetState();
                return false;
            }

            // 2. 取置信度最高的目标
            const DetectResult* target = &aiResults[0];
            for (const auto& res : aiResults)
            {
                if (res.confidence > target->confidence) target = &res;
            }

            // 3. 关键点数量不足 → 重置
            if (target->keypoints.size() < 17)
            {
                resetState();
                return false;
            }

            const auto& kp = target->keypoints;

            // 4. 肩、髋可见性检查
            bool hasShoulder = kp[L_SHOULDER].confidence > KPT_CONF_THRESHOLD ||
                               kp[R_SHOULDER].confidence > KPT_CONF_THRESHOLD;
            bool hasHip = kp[L_HIP].confidence > KPT_CONF_THRESHOLD ||
                          kp[R_HIP].confidence > KPT_CONF_THRESHOLD;
            if (!hasShoulder || !hasHip)
            {
                resetState();
                return false;
            }

            // 5. 计算肩中点与髋中点（两侧都可见则取平均，否则取可见侧）
            float shoulderX = 0.0f, shoulderY = 0.0f;
            int shoulderCnt = 0;
            if (kp[L_SHOULDER].confidence > KPT_CONF_THRESHOLD) { shoulderX += kp[L_SHOULDER].x; shoulderY += kp[L_SHOULDER].y; ++shoulderCnt; }
            if (kp[R_SHOULDER].confidence > KPT_CONF_THRESHOLD) { shoulderX += kp[R_SHOULDER].x; shoulderY += kp[R_SHOULDER].y; ++shoulderCnt; }
            shoulderX /= shoulderCnt;
            shoulderY /= shoulderCnt;

            float hipX = 0.0f, hipY = 0.0f;
            int hipCnt = 0;
            if (kp[L_HIP].confidence > KPT_CONF_THRESHOLD) { hipX += kp[L_HIP].x; hipY += kp[L_HIP].y; ++hipCnt; }
            if (kp[R_HIP].confidence > KPT_CONF_THRESHOLD) { hipX += kp[R_HIP].x; hipY += kp[R_HIP].y; ++hipCnt; }
            hipX /= hipCnt;
            hipY /= hipCnt;

            // 6. 身体轴线（肩→髋）与垂直方向夹角
            float dx = hipX - shoulderX;
            float dy = hipY - shoulderY;
            float angle = std::atan2(std::fabs(dx), std::fabs(dy)) * 180.0 / 3.14159265358979;

            // 7. 计算髋部下坠速度（图像 y 向下，下坠时 hipY 增大 → 速度为正）
            auto currentTime = std::chrono::steady_clock::now();
            float velocityY = 0.0f;
            if (hasPreviousTarget_)
            {
                float dt = std::chrono::duration<float>(currentTime - lastTime_).count();
                if (dt > 0.001f)
                {
                    velocityY = (hipY - previousHipY_) / dt;
                }
            }

            // 更新历史状态
            hasPreviousTarget_ = true;
            previousHipY_ = hipY;
            lastTime_ = currentTime;

            // 8. 快速下坠事件检测（进入时序窗口）
            if (velocityY > FALL_VELOCITY_THRESHOLD)
            {
                fallEventPending_ = true;
                fallEventFrames_ = 0;
                LOG_TRACE("检测到快速下坠！髋部速度 {:.1f} px/s", velocityY);
            }

            // 9. 时序窗口维护：下坠事件超过窗口则失效
            if (fallEventPending_)
            {
                fallEventFrames_++;
                if (fallEventFrames_ > FALL_EVENT_WINDOW)
                {
                    fallEventPending_ = false;
                }
            }

            // 10. 躺倒判定（身体轴线角度）
            if (angle > FALL_ANGLE_THRESHOLD)
            {
                lieConfirmCount_++;
                LOG_TRACE("疑似摔倒！身体倾角 {:.1f}°，连续帧数 {}", angle, lieConfirmCount_);
            }
            else
            {
                if (lieConfirmCount_ > 0)
                {
                    LOG_INFO("目标姿态恢复，警报解除");
                }
                lieConfirmCount_ = 0;
            }

            // 11. 报警判定
            // 路径 A：快速下坠 + 连续躺倒确认（动态摔倒）
            bool dynamicFall = fallEventPending_ && lieConfirmCount_ >= CONFIRM_FRAMES_THRESHOLD;
            // 路径 B：持续躺倒（静态兜底，可能没抓到下坠瞬间）
            bool staticLie = lieConfirmCount_ >= STATIC_LIE_THRESHOLD;

            if (dynamicFall || staticLie)
            {
                int centerX = target->x + target->width / 2;
                int centerY = target->y + target->height / 2;
                LOG_WARN("触发报警！身体倾角 {:.1f}°，下坠速度 {:.1f} px/s，目标中心（{}, {}）",
                         angle, velocityY, centerX, centerY);

                outEvent.isFall = true;
                outEvent.triggerBoxX = centerX;
                outEvent.triggerBoxY = centerY;
                outEvent.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

                resetState();
                return true;
            }

            return false;
        }
    }
}
