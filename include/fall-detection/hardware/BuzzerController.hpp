#pragma once
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

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
                BuzzerController(int gpioPin = 138);
                ~BuzzerController();

                bool init();

                // 异步触发警报，不阻塞主线程
                void triggerAlarm(int durationMs = 3000);
            
            private:
                bool writeSysfs(const std::string& path, const std::string& value);
                void alarmWorkerLoop();

            private:
                int gpioPin_;
                std::atomic<bool> isInitialized_{false};

                // 线程控制与同步原语
                std::thread workerThread_;
                std::atomic<bool> isRunning_{false};
                std::atomic<bool> alarmTriggered_{false};
                std::atomic<int> alarmDuration_{0};
                std::mutex mtx_;
                std::condition_variable cv_;
        };
    }
}