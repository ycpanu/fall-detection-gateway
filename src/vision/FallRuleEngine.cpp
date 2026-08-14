#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <cmath>
#include <algorithm>

namespace fall_detection
{
    namespace vision
    {
        FallRuleEngine::FallRuleEngine() : hasPreviousTarget_(false), previousCenterY_(0), previousClassId_(-1), lieConfirmCount_(0)
        {
            lastTime_ = std::chrono::steady_clock::now();
        }

        bool FallRuleEngine::processFrame(const std::vector<DetectResult>& aiResults, AlertEvent& outEvent)
        {
            // 1. 如果当前帧没有检测到任何人，重置追踪状态
            if (aiResults.empty())
            {
                hasPreviousTarget_ = false;
                lieConfirmCount_ = 0;
                return false;
            }

            // 2. 锁定监控目标：找到画面中置信度最高的目标
            DetectResult target = aiResults[0];
            for (const auto& res : aiResults)
            {
                if (res.confidence > target.confidence)
                {
                    target = res;
                }
            }

            // 3. 提取目标的物理几何特征
            // 计算目标外接矩形的中心点 Y 坐标
            int currentCenterY = target.y + target.height / 2;
            int currentCenterX = target.x + target.width / 2;

            // 校验：宽 > 高，说明人体处于横向状态
            bool isAspectRatioFall = (target.width > target.height);

            // 4. 计算时间差与 Y 轴下坠速度
            auto currentTime = std::chrono::steady_clock::now();
            float velocityY = 0.0f;

            if (hasPreviousTarget_)
            {
                // 计算两帧之间的时间
                std::chrono::duration<float> timeDelta = currentTime - lastTime_;
                float dt = timeDelta.count();

                // 防止高并发下的除以零异常
                if (dt > 0.001f)
                {
                    // 计算 Y 轴位移差，并得出瞬时下坠速度
                    float distanceY = static_cast<float>(currentCenterY - previousCenterY_);
                    velocityY = distanceY / dt;
                }
            }

            // 5. 更新内部历史状态
            hasPreviousTarget_ = true;
            previousCenterY_ = currentCenterY;
            previousClassId_ = target.classId;
            lastTime_ = currentTime;

            // 6. 核心摔倒判定规则
            // 条件 1：AI 静态特征判定当前姿势为 lie 
            // 条件 2：物理几何特征满足高宽比
            if (target.classId == CLASS_LIE && isAspectRatioFall)
            {
                // 条件 3：如果伴随极速下坠，或者我们已经处于持续躺下读秒状态
                if (velocityY > FALL_VELOCITY_THRESHOLD || lieConfirmCount_ > 0)
                {
                    lieConfirmCount_++;
                    LOG_TRACE("貌似摔倒！当前连续确认帧数：{}，瞬时下坠速度：{:.2f} px/s",lieConfirmCount_, velocityY);

                }

                // 边缘兜底逻辑：即使没有捕抓到极速下坠瞬间，只要符合平躺且姿势异常，强制开始计数
                else if (lieConfirmCount_ == 0)
                {
                    lieConfirmCount_++;
                }

                // 最终判定：如果连续 N 帧都确认时 lie 状态
                if (lieConfirmCount_ >= CONFIRM_FRAMES_THRESHOLD)
                {
                    LOG_WARN("触发报警！目标中心坐标：（{}, {}）", currentCenterX, currentCenterY);

                    // 封装报警事件数据，准备上报 MQTT
                    outEvent.isFall = true;
                    outEvent.triggerBoxX = currentCenterX;
                    outEvent.triggerBoxY = currentCenterY;
                    outEvent.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    
                    // 重置计数器
                    lieConfirmCount_ = 0;
                    return true;
                }
            }
            else
            {
                // 状态突变解除：如果当前帧不是 lie，或者人姿势变化，瞬间清零防抖计数器
                if (lieConfirmCount_ > 0)
                {
                        LOG_INFO("目标状态恢复，警报解除");
                        lieConfirmCount_ = 0;
                }
            }

            // 如果未命中完整摔倒规则，默认返回安全状态
            outEvent.isFall = false;
            return false;
        }
    }
}