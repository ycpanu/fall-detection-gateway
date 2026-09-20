#include "fall-detection/event/EventManager.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <chrono>

namespace fall_detection
{
    namespace event
    {
        EventManager::EventManager(const std::string& deviceId) : deviceId_(deviceId)
        {
            LOG_INFO("[EventManager] 初始化完成，device_id={}", deviceId_);
        }

        AlertEvent EventManager::prepareEvent(AlertEvent event)
        {
            if (event.timestamp <= 0)
            {
                event.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            }

            // 统一设备唯一标识
            event.deviceId = deviceId_;

            // 尚未分配 event_id 时统一生成
            if (event.eventId.empty())
            {
                event.eventId = generateEventId(event.timestamp);
            }

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