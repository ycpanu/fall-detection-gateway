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
            writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/value", "0");
            writeSysfs("/sys/class/gpio/unexport", std::to_string(gpioPin_));
        }

        bool BuzzerController::writeSysfs(const std::string& path, const std::string& value)
        {
            std::ofstream fs(path);
            if (!fs.is_open()) return false;
            fs << value;
            fs.close();
            return true;
        }

        bool BuzzerController::init()
        {
            // 导出 GPIO 引脚
            if (!writeSysfs("/sys/class/gpio/export", std::to_string(gpioPin_)))
            {
                LOG_ERROR("无法导出 GPIO 引脚 {}（可能被占用或编号不存在）: {}", gpioPin_, std::strerror(errno));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待系统生成节点
            // 设置为输出模式
            if (!writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/direction", "out"))
            {
                LOG_ERROR("无法设置 GPIO 引脚 {} 为输出模式: {}", gpioPin_, std::strerror(errno));
                return false;
            }
            LOG_INFO("蜂鸣器容灾模块初始化成功，引脚: {}", gpioPin_);
            return true;
        }

        void BuzzerController::triggerAlarm(int durationMs) 
        {
            // 启动独立线程蜂鸣，避免阻塞网络重连逻辑
            std::thread([this, durationMs]() 
            {
                LOG_WARN("📢 触发本地物理蜂鸣器报警！");
                writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/value", "1");
                std::this_thread::sleep_for(std::chrono::milliseconds(durationMs));
                writeSysfs("/sys/class/gpio/gpio" + std::to_string(gpioPin_) + "/value", "0");
            }).detach();
        }
    }
}