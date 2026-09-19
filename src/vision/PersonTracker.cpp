#include "fall-detection/vision/PersonTracker.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <algorithm>
#include <cmath>
#include <limits>


namespace fall_detection
{
    namespace vision
    {
        PersonTracker::PersonTracker(
            const PersonTrackerConfig& config)
            : config_(config)
        {
            LOG_INFO(
                "[PersonTracker] 初始化："
                "IoU阈值={:.2f}, 中心距离阈值={:.2f}, "
                "目标保留={}ms",
                config_.iouThreshold,
                config_.centerDistanceThreshold,
                config_.maxMissingMs
            );
        }


        void PersonTracker::reset()
        {
            tracks_.clear();

            // Track ID 不重新从1开始，
            // 避免调试日志中不同目标重复使用同一个编号。
        }


        float PersonTracker::calculateIoU(
            const DetectResult& a,
            const DetectResult& b) const
        {
            const int x1 =
                std::max(a.x, b.x);

            const int y1 =
                std::max(a.y, b.y);

            const int x2 =
                std::min(
                    a.x + a.width,
                    b.x + b.width
                );

            const int y2 =
                std::min(
                    a.y + a.height,
                    b.y + b.height
                );


            if (x1 >= x2 || y1 >= y2)
            {
                return 0.0f;
            }


            const float intersection =
                static_cast<float>(
                    (x2 - x1) *
                    (y2 - y1)
                );


            const float areaA =
                static_cast<float>(
                    a.width * a.height
                );

            const float areaB =
                static_cast<float>(
                    b.width * b.height
                );


            const float unionArea =
                areaA + areaB - intersection;


            if (unionArea <= 0.0f)
            {
                return 0.0f;
            }


            return intersection / unionArea;
        }


        float PersonTracker::
            calculateNormalizedCenterDistance(
                const DetectResult& a,
                const DetectResult& b) const
        {
            const float centerAX =
                static_cast<float>(a.x)
                + static_cast<float>(a.width) / 2.0f;

            const float centerAY =
                static_cast<float>(a.y)
                + static_cast<float>(a.height) / 2.0f;


            const float centerBX =
                static_cast<float>(b.x)
                + static_cast<float>(b.width) / 2.0f;

            const float centerBY =
                static_cast<float>(b.y)
                + static_cast<float>(b.height) / 2.0f;


            const float dx =
                centerAX - centerBX;

            const float dy =
                centerAY - centerBY;


            const float distance =
                std::sqrt(
                    dx * dx +
                    dy * dy
                );


            /*
             * 使用两个人体框平均高度进行归一化。
             *
             * 人离摄像头近时框大，
             * 人离摄像头远时框小，
             * 这样阈值不直接依赖像素尺寸。
             */
            const float referenceHeight =
                std::max(
                    (
                        static_cast<float>(a.height)
                        +
                        static_cast<float>(b.height)
                    ) / 2.0f,
                    1.0f
                );


            return distance / referenceHeight;
        }


        void PersonTracker::removeExpiredTracks(
            const std::chrono::steady_clock::time_point& now)
        {
            tracks_.erase(
                std::remove_if(
                    tracks_.begin(),
                    tracks_.end(),

                    [&](const Track& track)
                    {
                        const auto missingMs =
                            std::chrono::duration_cast<
                                std::chrono::milliseconds
                            >(
                                now - track.lastSeen
                            ).count();


                        if (missingMs >
                            config_.maxMissingMs)
                        {
                            LOG_DEBUG(
                                "[PersonTracker] "
                                "Track {} 超时删除",
                                track.id
                            );

                            return true;
                        }


                        return false;
                    }
                ),

                tracks_.end()
            );
        }


        std::vector<TrackedPerson>
        PersonTracker::update(
            const std::vector<DetectResult>& detections)
        {
            const auto now =
                std::chrono::steady_clock::now();


            /*
             * 即使当前帧一个人都没检测到，
             * 也不能立即清空 Track。
             *
             * 这样可以容忍短暂遮挡或模型偶尔漏一帧。
             */
            removeExpiredTracks(now);


            std::vector<TrackedPerson> results;

            results.reserve(
                detections.size()
            );


            /*
             * 标记本帧哪些旧 Track 已经被匹配，
             * 防止两个人同时匹配到同一个 Track。
             */
            std::vector<bool> trackUsed(
                tracks_.size(),
                false
            );


            for (const auto& detection :
                 detections)
            {
                int bestTrackIndex = -1;

                float bestScore =
                    -std::numeric_limits<float>::infinity();


                for (std::size_t i = 0;
                     i < tracks_.size();
                     ++i)
                {
                    if (trackUsed[i])
                    {
                        continue;
                    }


                    const float iou =
                        calculateIoU(
                            detection,
                            tracks_[i].lastDetection
                        );


                    const float centerDistance =
                        calculateNormalizedCenterDistance(
                            detection,
                            tracks_[i].lastDetection
                        );


                    /*
                     * 只要：
                     *
                     * IoU足够大
                     * OR
                     * 中心移动距离足够小
                     *
                     * 就认为有匹配可能。
                     */
                    if (iou < config_.iouThreshold &&
                        centerDistance >
                            config_.
                            centerDistanceThreshold)
                    {
                        continue;
                    }


                    /*
                     * 简单匹配得分：
                     *
                     * IoU 越大越好；
                     * 中心距离越小越好。
                     */
                    const float score =
                        iou - 0.30f * centerDistance;


                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestTrackIndex =
                            static_cast<int>(i);
                    }
                }


                // =====================================
                // 找到了已有 Track
                // =====================================
                if (bestTrackIndex >= 0)
                {
                    auto& track =
                        tracks_[bestTrackIndex];


                    track.lastDetection =
                        detection;

                    track.lastSeen =
                        now;


                    trackUsed[
                        bestTrackIndex
                    ] = true;


                    results.push_back(
                        {
                            track.id,
                            detection
                        }
                    );
                }

                // =====================================
                // 当前人体无法匹配，创建新 Track
                // =====================================
                else
                {
                    Track newTrack;

                    newTrack.id =
                        nextTrackId_++;

                    newTrack.lastDetection =
                        detection;

                    newTrack.lastSeen =
                        now;


                    tracks_.push_back(
                        newTrack
                    );


                    /*
                     * tracks_ 增加了一个元素，
                     * trackUsed 也同步增加。
                     */
                    trackUsed.push_back(true);


                    results.push_back(
                        {
                            newTrack.id,
                            detection
                        }
                    );


                    LOG_DEBUG(
                        "[PersonTracker] "
                        "创建新 Track {}",
                        newTrack.id
                    );
                }
            }


            return results;
        }
    }
}