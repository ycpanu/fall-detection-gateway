#pragma once
#include <atomic>
#include <string>
#include <mutex>

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
            explicit EventManager(
                const std::string& deviceId,
                const std::string& deploymentArea,
                const std::string& videoOutputDir,
                int64_t fusionWindowMs = 10000
            );

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
            std::string deploymentArea_;

            // 防止同一毫秒产生多个事件导致 ID 重复
            std::atomic<unsigned long long> sequence_{0};

            std::mutex mutex_;

            AlertEvent activeEvent_;
            bool hasActiveEvent_ = false;

            int64_t fusionWindowMs_ = 10000;

            std::string videoOutputDir_;
        };
    }
}