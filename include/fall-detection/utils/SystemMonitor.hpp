#pragma once

#include <string>

namespace fall_detection
{
    namespace utils
    {
        class SystemMonitor
        {
        public:
            // 获取 CPU 使用率，失败返回 -1
            int getCpuUsage() const;

            int getMemoryUsage() const;

            // 获取指定文件系统的存储使用率，失败返回 -1
            int getStorageUsage(const std::string& path = "/") const;

        };
    }
}