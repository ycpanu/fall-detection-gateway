#pragma once
#include <thread>
#include <atomic>

namespace fall_detection
{
    namespace hardware
    {
        /**
         * @brief 蜂鸣器物理报警控制器
         * 通过 Linux Sysfs 操控开发板 GPIO 引脚
         */
        class BuzzerController
        {
            public:
                BuzzerController(int gpioPin = 73);
                ~BuzzerController();

                bool init();

                // 异步触发警报，不阻塞主线程
                void triggerAlarm(int durationMs = 3000);

            private:
                int gpioPin_;
                bool writeSysfs(const std::string& path, const std::string& value);
        };
    }
}