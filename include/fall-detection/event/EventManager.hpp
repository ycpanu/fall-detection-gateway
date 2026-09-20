#pragma once
#include <atomic>
#include <string>

#include "fall-detection/event/AlertEvent.hpp"

namespace fall_detection
{
    namespace event
    {
        /**
         * @brief 统一报警事件管理器
         */
        class EventManager
        {
        public:
            explicit EventManager(const std::string& deviceId);

            /**
             * @brief 对感知模块产生的事件进行统一处理
             */
            AlertEvent prepareEvent(AlertEvent event);

        private:
            /**
             * @brief 生成唯一事件 ID
             */
            std::string generateEventId(long long timestamp);

        private:
            std::string deviceId_;

            // 防止同一毫秒产生多个事件导致 ID 重复
            std::atomic<unsigned long long> sequence_{0};
        };
    }
}