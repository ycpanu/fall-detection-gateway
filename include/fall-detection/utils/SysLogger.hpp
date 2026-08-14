#pragma once
// 全局宏定义
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#define LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(fall_detection::utils::SysLogger::getInstance().getLogger(), __VA_ARGS__)
#define LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(fall_detection::utils::SysLogger::getInstance().getLogger(), __VA_ARGS__)
#define LOG_INFO(...)     SPDLOG_LOGGER_INFO(fall_detection::utils::SysLogger::getInstance().getLogger(), __VA_ARGS__)
#define LOG_WARN(...)     SPDLOG_LOGGER_WARN(fall_detection::utils::SysLogger::getInstance().getLogger(), __VA_ARGS__)
#define LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(fall_detection::utils::SysLogger::getInstance().getLogger(), __VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(fall_detection::utils::SysLogger::getInstance().getLogger(), __VA_ARGS__)

#include <memory>
#include <string>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

namespace fall_detection
{
    namespace utils
    {
        // 采用“单例模式”设计的日志类，保证全局只有一个日志实例
        class SysLogger
        {
            private:
                // 私有化构造和析构函数，防止外部随便 new 这个类
                SysLogger() = default;
                ~SysLogger() = default;

                // 禁用拷贝构造和复制操作符，确保全局变量唯一
                SysLogger(const SysLogger&) = delete;
                SysLogger& operator=(const SysLogger&) = delete;

                // spdlog 异步日志器智能指针
                std::shared_ptr<spdlog::async_logger> logger_;
            
            public:
                //获取全局唯一实例的静态方法
                static SysLogger& getInstance()
                {
                    static SysLogger instance;
                    return instance;
                }

                // 初始化日志系统（在 main 函数最开头调用）
                void init(const std::string& log_file_path = "logs/gateway.log");

                // 获取日志器对象，供下面的宏调用
                std::shared_ptr<spdlog::async_logger> getLogger()
                {
                    return logger_;
                }

        };
    }
}
