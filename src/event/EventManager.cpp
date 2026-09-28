#include <chrono>
#include <algorithm>
#include <cstdlib>

#include "fall-detection/event/EventManager.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace event
    {
        EventManager::EventManager(
            const std::string& deviceId,
            const std::string& deploymentArea,
            const std::string& videoOutputDir,
            int64_t fusionWindowMs)
            : deviceId_(deviceId),
            deploymentArea_(deploymentArea),
            videoOutputDir_(videoOutputDir),
            fusionWindowMs_(fusionWindowMs)
        {
            LOG_INFO("[EventManager] 初始化完成，device_id={}", deviceId_);
        }

        bool hasSource(const std::vector<EventSource>& sources, EventSource target)
        {
            return std::find(sources.begin(), sources.end(), target) != sources.end();
        }

        AlertEvent EventManager::prepareEvent(AlertEvent event)
        {
            if (event.timestamp <= 0)
            {
                event.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            }

            event.deviceId = deviceId_;
            event.deploymentArea = deploymentArea_;

            // 尚未分配 event_id 时统一生成
            if (event.eventId.empty())
            {
                event.eventId = generateEventId(event.timestamp);
            }

            // 判断是否是视觉触发
            bool incomingVision = hasSource(event.source, EventSource::VISION);
            bool incomingVoice = hasSource(event.source, EventSource::VOICE);

            event.captureVideo = incomingVision;

            // 加锁
            std::lock_guard<std::mutex> lock(mutex_);

            // 填设备基础信息
            event.deviceId = deviceId_;
            event.deploymentArea = deploymentArea_;
            event.status = EventStatus::NEW;

            // 判断是否可以和当前事件融合
            bool canFuse = false;

            if (hasActiveEvent_)
            {
                int64_t timeDiff =
                    std::llabs(event.timestamp - activeEvent_.timestamp);

                bool activeVision =
                    hasSource(
                        activeEvent_.source,
                        EventSource::VISION
                    );

                bool activeVoice =
                    hasSource(
                        activeEvent_.source,
                        EventSource::VOICE
                    );

                bool crossModal =
                    (incomingVision && activeVoice) ||
                    (incomingVoice && activeVision);

                canFuse =
                    timeDiff <= fusionWindowMs_ &&
                    crossModal;
            }

            // 如果可以融合
            if (canFuse)
            {
                // 复用第一次报警的 ID
                event.eventId = activeEvent_.eventId;

                // 报警类型永远保留第一次
                event.eventType = activeEvent_.eventType;

                // 时间也保留第一次报警时间
                event.timestamp = activeEvent_.timestamp;

                event.isFusionUpdate = true;

                // 合并来源
                event.source = activeEvent_.source;

                if (incomingVision &&
                    !hasSource(event.source, EventSource::VISION))
                {
                    event.source.push_back(EventSource::VISION);
                }

                if (incomingVoice &&
                    !hasSource(event.source, EventSource::VOICE))
                {
                    event.source.push_back(EventSource::VOICE);
                }

                // 当前没有语音信息，则保留之前的关键词
                if (event.keyword.empty())
                    event.keyword = activeEvent_.keyword;

                // 当前不是视觉事件，则保留之前的视觉信息
                if (!incomingVision)
                {
                    event.personTrackId =
                        activeEvent_.personTrackId;

                    event.triggerBoxX =
                        activeEvent_.triggerBoxX;

                    event.triggerBoxY =
                        activeEvent_.triggerBoxY;

                    event.videoPath =
                        activeEvent_.videoPath;
                }

                // 如果这一次视觉是后到的，现在才创建同一个 event_id 的视频
                if (incomingVision && event.videoPath.empty())
                {
                    event.videoPath =
                        videoOutputDir_ +
                        "/fall_" +
                        event.eventId +
                        ".mp4";
                }

                activeEvent_ = event;

                return event;
            }

            event.isFusionUpdate = false;

            if (incomingVision)
            {
                event.videoPath =
                    videoOutputDir_ +
                    "/fall_" +
                    event.eventId +
                    ".mp4";
            }

            activeEvent_ = event;
            hasActiveEvent_ = true;

            LOG_INFO("[EventManager] 创建报警事件：event_id={}, device_id={}, type={}", event.eventId, event.deviceId, toString(event.eventType));
            return event;
        }

        std::string EventManager::generateEventId(long long timestamp)
        {
            const auto seq = sequence_.fetch_add(1);

            return deviceId_ + "_" + std::to_string(timestamp) + "_" + std::to_string(seq);
        }
    }
}