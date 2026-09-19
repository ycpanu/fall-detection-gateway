#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>

namespace fall_detection
{
    namespace vision
    {
        FallRuleEngine::FallRuleEngine(const FallRuleConfig& config)
            : config_(config)
        {
            auto now = std::chrono::steady_clock::now();

            lastFrameTime_ = now;
            suspectedStartTime_ = now;
            lieStartTime_ = now;
            lastFastDropTime_ = now;

            LOG_INFO(
                "跌倒规则引擎初始化：angle={}°, recovery={}°, "
                "hipVelocity={:.2f} body/s, centerVelocity={:.2f} body/s, "
                "suspect={}ms, staticLie={}ms",
                config_.fallAngleThreshold,
                config_.recoveryAngleThreshold,
                config_.normalizedVelocityThreshold,
                config_.normalizedCenterVelocityThreshold,
                config_.suspectConfirmMs,
                config_.staticLieConfirmMs
            );
        }


        void FallRuleEngine::resetToNormal()
        {
            state_ = FallState::NORMAL;

            lieTimerActive_ = false;
            fastDropDetected_ = false;
        }


        void FallRuleEngine::resetState()
        {
            resetToNormal();

            hasPreviousTarget_ = false;

            previousHipY_ = 0.0f;
            previousCenterY_ = 0.0f;
            previousBodyHeight_ = 0.0f;

            lastFrameTime_ = std::chrono::steady_clock::now();
        }


        bool FallRuleEngine::processFrame(
            const std::vector<DetectResult>& aiResults,
            AlertEvent& outEvent)
        {
            // 每次进入函数先清空输出事件。
            // 只有真正确认跌倒时才设置为 true。
            outEvent = AlertEvent{};

            constexpr int L_SHOULDER = 5;
            constexpr int R_SHOULDER = 6;
            constexpr int L_HIP = 11;
            constexpr int R_HIP = 12;

            const auto now = std::chrono::steady_clock::now();


            // =========================================================
            // 1. 当前帧没有检测到人体
            // =========================================================
            if (aiResults.empty())
            {
                resetState();
                return false;
            }


            // =========================================================
            // 2. 当前阶段仍然选择置信度最高的人
            //
            // 多人跟踪会在后续 feature/multi-person-tracking 分支实现。
            // 当前先保证单人状态机稳定。
            // =========================================================
            const DetectResult* target = &aiResults.front();

            for (const auto& result : aiResults)
            {
                if (result.confidence > target->confidence)
                {
                    target = &result;
                }
            }


            // =========================================================
            // 3. 检查关键点是否完整
            // =========================================================
            if (target->keypoints.size() < 17)
            {
                resetState();
                return false;
            }

            const auto& kp = target->keypoints;


            // =========================================================
            // 4. 检查肩部和髋部是否至少有一侧有效
            // =========================================================
            const bool leftShoulderValid =
                kp[L_SHOULDER].confidence >= config_.kptConfThreshold;

            const bool rightShoulderValid =
                kp[R_SHOULDER].confidence >= config_.kptConfThreshold;

            const bool leftHipValid =
                kp[L_HIP].confidence >= config_.kptConfThreshold;

            const bool rightHipValid =
                kp[R_HIP].confidence >= config_.kptConfThreshold;


            if ((!leftShoulderValid && !rightShoulderValid) ||
                (!leftHipValid && !rightHipValid))
            {
                resetState();
                return false;
            }


            // =========================================================
            // 5. 计算肩部中点
            // =========================================================
            float shoulderX = 0.0f;
            float shoulderY = 0.0f;
            int shoulderCount = 0;

            if (leftShoulderValid)
            {
                shoulderX += kp[L_SHOULDER].x;
                shoulderY += kp[L_SHOULDER].y;
                ++shoulderCount;
            }

            if (rightShoulderValid)
            {
                shoulderX += kp[R_SHOULDER].x;
                shoulderY += kp[R_SHOULDER].y;
                ++shoulderCount;
            }

            shoulderX /= static_cast<float>(shoulderCount);
            shoulderY /= static_cast<float>(shoulderCount);


            // =========================================================
            // 6. 计算髋部中点
            // =========================================================
            float hipX = 0.0f;
            float hipY = 0.0f;
            int hipCount = 0;

            if (leftHipValid)
            {
                hipX += kp[L_HIP].x;
                hipY += kp[L_HIP].y;
                ++hipCount;
            }

            if (rightHipValid)
            {
                hipX += kp[R_HIP].x;
                hipY += kp[R_HIP].y;
                ++hipCount;
            }

            hipX /= static_cast<float>(hipCount);
            hipY /= static_cast<float>(hipCount);


            // =========================================================
            // 7. 计算人体躯干倾角
            //
            // 图像坐标：
            // X 向右
            // Y 向下
            //
            // 与垂直方向夹角：
            //
            // 站立 ≈ 0°
            // 水平躺倒 ≈ 90°
            // =========================================================
            const float dx = hipX - shoulderX;
            const float dy = hipY - shoulderY;

            const float angle =
                std::atan2(std::abs(dx), std::abs(dy))
                * 180.0f
                / static_cast<float>(M_PI);


            // =========================================================
            // 8. 获取人体 Bounding Box 信息
            // =========================================================
            const float bodyHeight =
                static_cast<float>(std::max(target->height, 1));

            const float centerX =
                static_cast<float>(target->x)
                + static_cast<float>(target->width) / 2.0f;

            const float centerY =
                static_cast<float>(target->y)
                + bodyHeight / 2.0f;


            // =========================================================
            // 9. 计算归一化运动速度
            //
            // normalizedHipVelocity:
            //
            //           髋部Y位移
            // --------------------------------
            //      人体高度 × 时间
            //
            // 单位可理解为：
            // body-height / second
            // =========================================================
            float normalizedHipVelocity = 0.0f;
            float normalizedCenterVelocity = 0.0f;


            if (hasPreviousTarget_)
            {
                const float dt =
                    std::chrono::duration<float>(
                        now - lastFrameTime_
                    ).count();

                /*
                 * 如果两帧之间时间异常，例如程序暂停了一两秒，
                 * 就不使用这两帧计算运动速度。
                 *
                 * 否则可能把程序卡顿误认为人体突然移动。
                 */
                if (dt > 0.001f && dt < 1.0f)
                {
                    /*
                     * 使用当前帧与上一帧人体高度平均值作为尺度。
                     *
                     * 防止人体框本身有轻微抖动时，
                     * 归一化结果变化过大。
                     */
                    const float referenceHeight =
                        std::max(
                            (bodyHeight + previousBodyHeight_) / 2.0f,
                            1.0f
                        );


                    normalizedHipVelocity =
                        (hipY - previousHipY_)
                        / referenceHeight
                        / dt;


                    normalizedCenterVelocity =
                        (centerY - previousCenterY_)
                        / referenceHeight
                        / dt;
                }
            }


            // =========================================================
            // 10. 判断当前基本特征
            // =========================================================

            // 严重倾斜 / 躺倒
            const bool isLying =
                angle >= config_.fallAngleThreshold;


            // 已明显恢复站立
            //
            // recoveryAngleThreshold 比 fallAngleThreshold 小，
            // 形成“滞回区”，防止状态在阈值附近频繁跳变。
            const bool isRecovered =
                angle <= config_.recoveryAngleThreshold;


            const bool hipFastDrop =
                normalizedHipVelocity >=
                config_.normalizedVelocityThreshold;


            const bool centerFastDrop =
                normalizedCenterVelocity >=
                config_.normalizedCenterVelocityThreshold;


            // 两种运动证据满足一种即可认为发生明显下降
            const bool fastDrop =
                hipFastDrop || centerFastDrop;


            // =========================================================
            // 11. 记录快速下降事件
            // =========================================================
            if (fastDrop)
            {
                fastDropDetected_ = true;
                lastFastDropTime_ = now;

                LOG_DEBUG(
                    "[FallRule] 检测到快速下降："
                    "hip={:.2f} body/s, "
                    "center={:.2f} body/s, "
                    "angle={:.1f}°",
                    normalizedHipVelocity,
                    normalizedCenterVelocity,
                    angle
                );
            }


            // 快速下降事件超过有效窗口后失效
            if (fastDropDetected_)
            {
                const auto elapsed =
                    std::chrono::duration_cast<
                        std::chrono::milliseconds
                    >(now - lastFastDropTime_).count();

                if (elapsed > config_.fallEventWindowMs)
                {
                    fastDropDetected_ = false;
                }
            }


            // =========================================================
            // 12. 静态躺卧计时器
            // =========================================================
            if (isLying)
            {
                if (!lieTimerActive_)
                {
                    lieTimerActive_ = true;
                    lieStartTime_ = now;
                }
            }
            else
            {
                lieTimerActive_ = false;
            }


            // =========================================================
            // 13. 状态机
            // =========================================================

            switch (state_)
            {
                // -----------------------------------------------------
                // NORMAL
                // -----------------------------------------------------
                case FallState::NORMAL:
                {
                    /*
                     * 动态跌倒入口：
                     *
                     * 最近检测到明显下降
                     * +
                     * 身体已经出现明显倾斜
                     *
                     * recoveryAngleThreshold 在这里作为“已经明显偏离站立”
                     * 的最低要求。
                     */
                    if (fastDropDetected_ &&
                        angle > config_.recoveryAngleThreshold)
                    {
                        state_ = FallState::SUSPECTED_FALL;
                        suspectedStartTime_ = now;

                        LOG_INFO(
                            "[FallRule] NORMAL -> SUSPECTED_FALL "
                            "(angle={:.1f}°, hipV={:.2f}, centerV={:.2f})",
                            angle,
                            normalizedHipVelocity,
                            normalizedCenterVelocity
                        );
                    }

                    /*
                     * 静态兜底：
                     *
                     * 没抓到快速下降，
                     * 但人体长时间保持水平躺倒。
                     *
                     * 下一阶段加入安全区域以后，
                     * 床和沙发上的正常躺卧将在这里被过滤。
                     */
                    else if (isLying && lieTimerActive_)
                    {
                        const auto lieDuration =
                            std::chrono::duration_cast<
                                std::chrono::milliseconds
                            >(now - lieStartTime_).count();

                        if (lieDuration >=
                            config_.staticLieConfirmMs)
                        {
                            state_ = FallState::CONFIRMED_FALL;

                            LOG_WARN(
                                "[FallRule] 静态异常躺卧确认："
                                "{} ms，进入 CONFIRMED_FALL",
                                lieDuration
                            );
                        }
                    }

                    break;
                }


                // -----------------------------------------------------
                // SUSPECTED_FALL
                // -----------------------------------------------------
                case FallState::SUSPECTED_FALL:
                {
                    /*
                     * 疑似跌倒以后如果快速恢复，
                     * 认为可能只是快速弯腰、坐下等动作。
                     */
                    if (isRecovered)
                    {
                        LOG_INFO(
                            "[FallRule] 疑似跌倒后恢复正常，"
                            "SUSPECTED_FALL -> NORMAL"
                        );

                        resetToNormal();
                        break;
                    }


                    const auto suspectedDuration =
                        std::chrono::duration_cast<
                            std::chrono::milliseconds
                        >(now - suspectedStartTime_).count();


                    /*
                     * 疑似跌倒后持续处于躺倒姿态达到确认时间，
                     * 才真正确认跌倒。
                     */
                    if (isLying &&
                        suspectedDuration >=
                        config_.suspectConfirmMs)
                    {
                        state_ = FallState::CONFIRMED_FALL;

                        LOG_WARN(
                            "[FallRule] 动态跌倒确认："
                            "angle={:.1f}°, duration={} ms",
                            angle,
                            suspectedDuration
                        );
                    }

                    /*
                     * 如果已经超过跌落事件窗口，
                     * 又始终没有形成躺倒状态，
                     * 本次疑似事件取消。
                     */
                    else if (
                        suspectedDuration >
                        config_.fallEventWindowMs)
                    {
                        LOG_INFO(
                            "[FallRule] 疑似跌倒超时未确认，"
                            "SUSPECTED_FALL -> NORMAL"
                        );

                        resetToNormal();
                    }

                    break;
                }


                // -----------------------------------------------------
                // CONFIRMED_FALL
                // -----------------------------------------------------
                case FallState::CONFIRMED_FALL:
                {
                    /*
                     * CONFIRMED_FALL 是一个很短暂的中间状态。
                     *
                     * 在这里生成一次报警事件，
                     * 随后立即进入 ALARMED。
                     */
                    outEvent.isFall = true;

                    outEvent.timestamp =
                        std::chrono::duration_cast<
                            std::chrono::milliseconds
                        >(
                            std::chrono::system_clock::now()
                                .time_since_epoch()
                        ).count();


                    outEvent.triggerBoxX =
                        static_cast<int>(centerX);

                    outEvent.triggerBoxY =
                        static_cast<int>(centerY);


                    state_ = FallState::ALARMED;

                    LOG_WARN(
                        "[FallRule] CONFIRMED_FALL -> ALARMED，"
                        "触发报警，位置=({}, {})",
                        outEvent.triggerBoxX,
                        outEvent.triggerBoxY
                    );


                    // 更新当前人体历史数据后返回
                    previousHipY_ = hipY;
                    previousCenterY_ = centerY;
                    previousBodyHeight_ = bodyHeight;
                    lastFrameTime_ = now;
                    hasPreviousTarget_ = true;

                    return true;
                }


                // -----------------------------------------------------
                // ALARMED
                // -----------------------------------------------------
                case FallState::ALARMED:
                {
                    /*
                     * 已经报警以后，只要人体仍处于倒地状态，
                     * 就不重复报警。
                     */
                    if (isRecovered)
                    {
                        LOG_INFO(
                            "[FallRule] 人体已恢复正常姿态，"
                            "ALARMED -> NORMAL"
                        );

                        resetToNormal();
                    }

                    break;
                }
            }


            // =========================================================
            // 14. 如果本帧刚刚从 NORMAL/SUSPECTED 进入
            //     CONFIRMED_FALL，则立即生成事件
            //
            // 因为 switch 已经执行过 CONFIRMED case，
            // 需要在这里处理本帧刚转换的情况。
            // =========================================================
            if (state_ == FallState::CONFIRMED_FALL)
            {
                outEvent.isFall = true;

                outEvent.timestamp =
                    std::chrono::duration_cast<
                        std::chrono::milliseconds
                    >(
                        std::chrono::system_clock::now()
                            .time_since_epoch()
                    ).count();

                outEvent.triggerBoxX =
                    static_cast<int>(centerX);

                outEvent.triggerBoxY =
                    static_cast<int>(centerY);

                state_ = FallState::ALARMED;

                LOG_WARN(
                    "[FallRule] 跌倒报警触发，位置=({}, {})",
                    outEvent.triggerBoxX,
                    outEvent.triggerBoxY
                );


                previousHipY_ = hipY;
                previousCenterY_ = centerY;
                previousBodyHeight_ = bodyHeight;
                lastFrameTime_ = now;
                hasPreviousTarget_ = true;

                return true;
            }


            // =========================================================
            // 15. 保存当前帧运动信息，供下一帧计算速度
            // =========================================================
            previousHipY_ = hipY;
            previousCenterY_ = centerY;
            previousBodyHeight_ = bodyHeight;

            lastFrameTime_ = now;
            hasPreviousTarget_ = true;


            // 调试阶段需要时可将 log_level 设置为 DEBUG
            LOG_TRACE(
                "[FallRule] state={}, angle={:.1f}, "
                "hipV={:.2f}, centerV={:.2f}, lying={}",
                static_cast<int>(state_),
                angle,
                normalizedHipVelocity,
                normalizedCenterVelocity,
                isLying
            );

            return false;
        }
    }
}