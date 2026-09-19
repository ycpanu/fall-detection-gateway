#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace fall_detection
{
    namespace vision
    {
        FallRuleEngine::FallRuleEngine(const FallRuleConfig& config, const SafeZoneManager* safeZoneManager) : config_(config), safeZoneManager_(safeZoneManager)
        {
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


        void FallRuleEngine::resetPersonToNormal(PersonState& personState)
        {
            personState.state =
                FallState::NORMAL;

            personState.lieTimerActive =
                false;

            personState.fastDropDetected =
                false;
        }

        void FallRuleEngine::removeExpiredPersonStates(
            const std::chrono::steady_clock::time_point& now)
        {
            /*
            * PersonTracker 当前短暂保留 Track 约 1.5 秒。
            *
            * FallRuleEngine 再稍微多保留一点时间，
            * 避免短暂遮挡直接丢失跌倒状态。
            */
            constexpr int STATE_RETENTION_MS = 2500;


            for (auto it = personStates_.begin();
                it != personStates_.end();)
            {
                const auto missingMs =
                    std::chrono::duration_cast<
                        std::chrono::milliseconds
                    >(
                        now - it->second.lastSeenTime
                    ).count();


                if (missingMs > STATE_RETENTION_MS)
                {
                    LOG_DEBUG(
                        "[FallRule] Track {} 状态超时清除",
                        it->first
                    );

                    it =
                        personStates_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        std::vector<AlertEvent> FallRuleEngine::processFrame(const std::vector<TrackedPerson>& trackedPersons, int frameWidth, int frameHeight)
        {
            const auto now =
                std::chrono::steady_clock::now();


            // 清理已经长时间消失的人
            removeExpiredPersonStates(now);


            std::vector<AlertEvent> events;


            /*
            * 每个人分别进入自己的跌倒状态机。
            */
            for (const auto& person :
                trackedPersons)
            {
                /*
                * 如果该 Track 第一次出现，
                * unordered_map 会自动创建一份新的 PersonState。
                */
                auto& personState =
                    personStates_[person.trackId];


                personState.lastSeenTime =
                    now;


                AlertEvent event;


                if (processPerson(
                        person,
                        frameWidth,
                        frameHeight,
                        personState,
                        event))
                {
                    events.push_back(
                        std::move(event)
                    );
                }
            }


            return events;
        }

        bool FallRuleEngine::processPerson(const TrackedPerson& person, int frameWidth,
            int frameHeight, PersonState& personState, AlertEvent& outEvent)
        {
            outEvent = AlertEvent{};

            const auto& target =
                person.detection;

            const int trackId =
                person.trackId;


            constexpr int L_SHOULDER = 5;
            constexpr int R_SHOULDER = 6;
            constexpr int L_HIP = 11;
            constexpr int R_HIP = 12;


            const auto now =
                std::chrono::steady_clock::now();


            // =====================================================
            // 关键点检查
            // =====================================================
            if (target.keypoints.size() < 17)
            {
                /*
                * 这里不能 reset。
                *
                * 某一帧关键点识别失败，
                * 保留这个人的历史状态。
                */
                return false;
            }


            const auto& kp =
                target.keypoints;


            const bool leftShoulderValid =
                kp[L_SHOULDER].confidence >=
                config_.kptConfThreshold;

            const bool rightShoulderValid =
                kp[R_SHOULDER].confidence >=
                config_.kptConfThreshold;

            const bool leftHipValid =
                kp[L_HIP].confidence >=
                config_.kptConfThreshold;

            const bool rightHipValid =
                kp[R_HIP].confidence >=
                config_.kptConfThreshold;


            if ((!leftShoulderValid &&
                !rightShoulderValid) ||
                (!leftHipValid &&
                !rightHipValid))
            {
                return false;
            }


            // =====================================================
            // 肩部中点
            // =====================================================
            float shoulderX = 0.0f;
            float shoulderY = 0.0f;
            int shoulderCount = 0;


            if (leftShoulderValid)
            {
                shoulderX +=
                    kp[L_SHOULDER].x;

                shoulderY +=
                    kp[L_SHOULDER].y;

                ++shoulderCount;
            }


            if (rightShoulderValid)
            {
                shoulderX +=
                    kp[R_SHOULDER].x;

                shoulderY +=
                    kp[R_SHOULDER].y;

                ++shoulderCount;
            }


            shoulderX /=
                static_cast<float>(
                    shoulderCount
                );

            shoulderY /=
                static_cast<float>(
                    shoulderCount
                );


            // =====================================================
            // 髋部中点
            // =====================================================
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


            hipX /=
                static_cast<float>(
                    hipCount
                );

            hipY /=
                static_cast<float>(
                    hipCount
                );


            // =====================================================
            // 身体倾角
            // =====================================================
            const float dx =
                hipX - shoulderX;

            const float dy =
                hipY - shoulderY;


            const float angle =
                std::atan2(
                    std::abs(dx),
                    std::abs(dy)
                )
                *
                180.0f
                /
                static_cast<float>(M_PI);


            // =====================================================
            // Bounding Box
            // =====================================================
            const float bodyHeight =
                static_cast<float>(
                    std::max(target.height, 1)
                );


            const float centerX =
                static_cast<float>(target.x)
                +
                static_cast<float>(
                    target.width
                ) / 2.0f;


            const float centerY =
                static_cast<float>(target.y)
                +
                bodyHeight / 2.0f;


            // =====================================================
            // 运动速度
            // =====================================================
            float normalizedHipVelocity =
                0.0f;

            float normalizedCenterVelocity =
                0.0f;


            if (personState.hasPreviousTarget)
            {
                const float dt =
                    std::chrono::duration<float>(
                        now -
                        personState.lastFrameTime
                    ).count();


                if (dt > 0.001f &&
                    dt < 1.0f)
                {
                    const float referenceHeight =
                        std::max(
                            (
                                bodyHeight +
                                personState.
                                    previousBodyHeight
                            ) / 2.0f,
                            1.0f
                        );


                    normalizedHipVelocity =
                        (
                            hipY -
                            personState.previousHipY
                        )
                        /
                        referenceHeight
                        /
                        dt;


                    normalizedCenterVelocity =
                        (
                            centerY -
                            personState.previousCenterY
                        )
                        /
                        referenceHeight
                        /
                        dt;
                }
            }


            // =====================================================
            // 安全区域
            // =====================================================
            bool inSafeLieZone = false;
            float overlapRatio = 0.0f;


            if (safeZoneManager_ != nullptr &&
                frameWidth > 0 &&
                frameHeight > 0)
            {
                const float normalizedHipX =
                    std::clamp(
                        hipX /
                        static_cast<float>(
                            frameWidth
                        ),
                        0.0f,
                        1.0f
                    );


                const float normalizedHipY =
                    std::clamp(
                        hipY /
                        static_cast<float>(
                            frameHeight
                        ),
                        0.0f,
                        1.0f
                    );


                const bool hipInside =
                    safeZoneManager_->
                        isPointInSafeZone(
                            normalizedHipX,
                            normalizedHipY
                        );


                overlapRatio =
                    safeZoneManager_->
                        calculateBoxOverlapRatio(
                            target.x,
                            target.y,
                            target.width,
                            target.height,
                            frameWidth,
                            frameHeight
                        );


                inSafeLieZone =
                    hipInside &&
                    overlapRatio >=
                        config_.
                        safeZoneOverlapThreshold;
            }


            // =====================================================
            // 基础特征
            // =====================================================
            const bool isLying =
                angle >=
                config_.fallAngleThreshold;


            const bool isRecovered =
                angle <=
                config_.recoveryAngleThreshold;


            const bool hipFastDrop =
                normalizedHipVelocity >=
                config_.
                    normalizedVelocityThreshold;


            const bool centerFastDrop =
                normalizedCenterVelocity >=
                config_.
                    normalizedCenterVelocityThreshold;


            const bool fastDrop =
                hipFastDrop ||
                centerFastDrop;


            // =====================================================
            // 快速下降时间窗口
            // =====================================================
            if (fastDrop)
            {
                personState.fastDropDetected =
                    true;

                personState.lastFastDropTime =
                    now;
            }


            if (personState.fastDropDetected)
            {
                const auto elapsed =
                    std::chrono::duration_cast<
                        std::chrono::milliseconds
                    >(
                        now -
                        personState.lastFastDropTime
                    ).count();


                if (elapsed >
                    config_.fallEventWindowMs)
                {
                    personState.fastDropDetected =
                        false;
                }
            }


            // =====================================================
            // 静态躺卧计时
            // =====================================================
            if (isLying &&
                !inSafeLieZone)
            {
                if (!personState.lieTimerActive)
                {
                    personState.lieTimerActive =
                        true;

                    personState.lieStartTime =
                        now;
                }
            }
            else
            {
                personState.lieTimerActive =
                    false;
            }


            bool confirmedThisFrame =
                false;


            // =====================================================
            // 每个人自己的状态机
            // =====================================================
            switch (personState.state)
            {
                case FallState::NORMAL:
                {
                    if (
                        personState.fastDropDetected &&
                        angle >
                            config_.
                            recoveryAngleThreshold)
                    {
                        personState.state =
                            FallState::SUSPECTED_FALL;

                        personState.
                            suspectedStartTime =
                            now;


                        LOG_INFO(
                            "[FallRule][Track {}] "
                            "NORMAL -> SUSPECTED_FALL "
                            "angle={:.1f}, hipV={:.2f}, "
                            "centerV={:.2f}",
                            trackId,
                            angle,
                            normalizedHipVelocity,
                            normalizedCenterVelocity
                        );
                    }
                    else if (
                        isLying &&
                        !inSafeLieZone &&
                        personState.lieTimerActive)
                    {
                        const auto lieDuration =
                            std::chrono::duration_cast<
                                std::chrono::milliseconds
                            >(
                                now -
                                personState.lieStartTime
                            ).count();


                        if (lieDuration >=
                            config_.
                            staticLieConfirmMs)
                        {
                            personState.state =
                                FallState::
                                CONFIRMED_FALL;

                            confirmedThisFrame =
                                true;


                            LOG_WARN(
                                "[FallRule][Track {}] "
                                "静态异常躺卧确认",
                                trackId
                            );
                        }
                    }

                    break;
                }


                case FallState::SUSPECTED_FALL:
                {
                    if (isRecovered)
                    {
                        LOG_INFO(
                            "[FallRule][Track {}] "
                            "SUSPECTED_FALL -> NORMAL",
                            trackId
                        );


                        resetPersonToNormal(
                            personState
                        );

                        break;
                    }


                    const auto duration =
                        std::chrono::duration_cast<
                            std::chrono::milliseconds
                        >(
                            now -
                            personState.
                                suspectedStartTime
                        ).count();


                    if (
                        isLying &&
                        duration >=
                            config_.suspectConfirmMs)
                    {
                        personState.state =
                            FallState::
                            CONFIRMED_FALL;

                        confirmedThisFrame =
                            true;
                    }
                    else if (
                        duration >
                        config_.fallEventWindowMs)
                    {
                        resetPersonToNormal(
                            personState
                        );
                    }

                    break;
                }


                case FallState::CONFIRMED_FALL:
                {
                    confirmedThisFrame =
                        true;

                    break;
                }


                case FallState::ALARMED:
                {
                    if (isRecovered)
                    {
                        LOG_INFO(
                            "[FallRule][Track {}] "
                            "ALARMED -> NORMAL",
                            trackId
                        );


                        resetPersonToNormal(
                            personState
                        );
                    }

                    break;
                }
            }


            // =====================================================
            // 生成该人的报警
            // =====================================================
            if (confirmedThisFrame)
            {
                outEvent.isFall = true;

                outEvent.personTrackId =
                    trackId;


                outEvent.timestamp =
                    std::chrono::duration_cast<
                        std::chrono::milliseconds
                    >(
                        std::chrono::
                            system_clock::now()
                            .time_since_epoch()
                    ).count();


                outEvent.triggerBoxX =
                    static_cast<int>(
                        centerX
                    );

                outEvent.triggerBoxY =
                    static_cast<int>(
                        centerY
                    );


                personState.state =
                    FallState::ALARMED;


                LOG_WARN(
                    "[FallRule][Track {}] "
                    "确认跌倒，进入 ALARMED",
                    trackId
                );
            }


            // =====================================================
            // 保存该人的当前数据
            // =====================================================
            personState.previousHipY =
                hipY;

            personState.previousCenterY =
                centerY;

            personState.previousBodyHeight =
                bodyHeight;

            personState.lastFrameTime =
                now;

            personState.hasPreviousTarget =
                true;


            return confirmedThisFrame;
        }
    }
}