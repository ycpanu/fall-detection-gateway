#include "fall-detection/hardware/BuzzerController.hpp"
#include "fall-detection/utils/SysLogger.hpp"
#include <fstream>
#include <chrono>
#include <cerrno>
#include <cstring>

namespace fall_detection
{
    namespace hardware
    {
        BuzzerController::BuzzerController(int gpioPin) : gpioPin_(gpioPin) {}
        BuzzerController::~BuzzerController()
        {
            // 1. 停止后台线程
            if (isRunning_)
            {
                isRunning_ = false;
                cv_.notify_one();
                if (workerThread_.joinable())
                {
                    workerThread_.join();
                }
            }

            // 2. 释放硬件资源
            if (isInitialized_)
            {
                writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/value", "0");
                writeSysfs("/sys/class/gpio/unexpoert", std::to_string(gpioPin_));
                LOG_INFO("蜂鸣器资源已安全释放。");
            }
        }

        bool BuzzerController::writeSysfs(const std::string& path, const std::string& value)
        {
            std::ofstream fs(path);
            if (!fs.is_open()) return false;
            fs << value;
            return fs.good();
        }

        bool BuzzerController::init()
        {
            if (isInitialized_) return true;

            if (!writeSysfs("/sys/class/gpio/export", std::to_string(gpioPin_)))
            {
                LOG_ERROR("无法导出 GPIO 引脚 {} : {}", gpioPin_, std::strerror(errno));
                return false;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (!writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/direction", "out"))
            {
                LOG_ERROR("无法设置 GPIO 引脚 {} 为输出模式 : {}", gpioPin_, std::strerror(errno));
                return false;
            }
            LOG_INFO("蜂鸣器容灾模块初始化成功，引脚: {}", gpioPin_);
            isInitialized_ = true;
            isRunning_ = true;
            return true;

            // 启动常驻工作线程，彻底消灭 detach 游离线程
            workerThread_ = std::thread(&BuzzerController::alarmWorkerLoop, this);
            
            LOG_INFO("蜂鸣器容灾模块初始化成功，引脚: {}", gpioPin_);
            return true;
        }

        void BuzzerController::triggerAlarm(int durationMs) 
        {
            if (!isInitialized_) return;
            
            // 仅更新状态并唤醒工作线程，绝不阻塞主流水线
            {
                std::lock_guard<std::mutex> lock(mtx_);
                alarmDuration_ = durationMs;
                alarmTriggered_ = true;
            }
            cv_.notify_one();
        }

        void BuzzerController::alarmWorkerLoop()
        {
            while(isRunning_)
            {
                std::unique_lock<std::mutex> lock(mtx_);
                
                // 阻塞休眠，零 CPU 占用，等待触发警报或线程终止信号
                cv_.wait(lock, [this]() { return alarmTriggered_ || !isRunning_; });

                if (!isRunning_) break; // 线程终止信号，退出循环

                // 消耗掉触发信号，避免重复触发
                alarmTriggered_ = false;
                int duration = alarmDuration_.load();
                lock.unlock();  // 释放锁，避免休眠期间占用互斥锁

                LOG_WARN("蜂鸣器警报触发，持续时间: {} ms", duration);
                writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/value", "1");

                // 响应期间若系统要求退出，立即停止蜂鸣器
                auto startTime = std::chrono::steady_clock::now();
                while (isRunning_ && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count() < duration)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/value", "0");
                LOG_INFO("蜂鸣器警报结束。");
            }
        }
    }
}