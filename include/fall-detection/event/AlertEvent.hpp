#pragma once

#include <string>
#include <vector>

namespace fall_detection
{
    namespace event
    {
        /**
         * @brief 报警事件类型
         */
        enum class EventType
        {
            FALL,
            HELP_REQUEST
        };

        /**
         * @brief 事件检测来源
         */
        enum class EventSource
        {
            VISION,
            VOICE
        };

        /**
         * @brief 人工处理状态
         */
        enum class EventStatus
        {
            NEW,
            ACKNOWLEDGED,
            RESOLVED
        };

        /**
         * @brief 统一报警事件
         */
        struct AlertEvent
        {
            std::string eventId;
            std::string deviceId;
            std::string deploymentArea;
            EventType eventType = EventType::FALL;

            // 可同时包含 vision + voice
            std::vector<EventSource> source;

            EventStatus status = EventStatus::NEW;

            // 事件发生时间, Unix 毫秒
            long long timestamp = 0;
            
            // 视觉人体临时 Track ID
            int personTrackId = -1;

            int triggerBoxX = 0;
            int triggerBoxY = 0;

            // 语音求救关键词
            std::string keyword;

            std::string videoPath;
        };

        // 枚举转字符串
        inline std::string toString(EventType type)
        {
            switch (type)
            {
                case EventType::FALL:
                    return "FALL";

                case EventType::HELP_REQUEST:
                    return "HELP_REQUEST";
                
            }
            return "UNKNOWN";
        }

        inline std::string toString(EventSource source)
        {
            switch (source)
            {
                case EventSource::VISION:
                    return "VISION";

                case EventSource::VOICE:
                    return "VOICE";
            }

            return "UNKNOWN";
        }

        inline std::string toString(EventStatus status)
        {
            switch (status)
            {
                case EventStatus::NEW:
                    return "NEW";

                case EventStatus::ACKNOWLEDGED:
                    return "ACKNOWLEDGED";

                case EventStatus::RESOLVED:
                    return "RESOLVED";
            }

            return "UNKNOWN";
        }
    }
}