#include "fall-detection/utils/SystemMonitor.hpp"

#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <sys/statvfs.h>

namespace fall_detection
{
    namespace utils
    {
        namespace
        {
            bool readCpuTimes(unsigned long long& idle, unsigned long long& total)
            {
                std::ifstream file("/proc/stat");
                if (!file.is_open())
                    return false;

                std::string line;
                std::getline(file, line);

                std::istringstream iss(line);

                std::string cpu;
                unsigned long long user = 0;
                unsigned long long nice = 0;
                unsigned long long system = 0;
                unsigned long long idleTime = 0;
                unsigned long long iowait = 0;
                unsigned long long irq = 0;
                unsigned long long softirq = 0;
                unsigned long long steal = 0;

                iss >> cpu
                    >> user
                    >> nice
                    >> system
                    >> idleTime
                    >> iowait
                    >> irq
                    >> softirq
                    >> steal;

                if (cpu != "cpu")
                    return false;

                idle = idleTime + iowait;

                total =
                    user +
                    nice +
                    system +
                    idleTime +
                    iowait +
                    irq +
                    softirq +
                    steal;

                return true;
            }
        }

        int SystemMonitor::getCpuUsage() const
        {
            unsigned long long idle1 = 0;
            unsigned long long total1 = 0;
            unsigned long long idle2 = 0;
            unsigned long long total2 = 0;

            if (!readCpuTimes(idle1, total1))
                return -1;

            std::this_thread::sleep_for(
                std::chrono::milliseconds(200)
            );

            if (!readCpuTimes(idle2, total2))
                return -1;

            unsigned long long totalDelta = total2 - total1;
            unsigned long long idleDelta = idle2 - idle1;

            if (totalDelta == 0)
                return -1;

            double usage =
                100.0 *
                static_cast<double>(totalDelta - idleDelta) /
                static_cast<double>(totalDelta);

            if (usage < 0.0)
                usage = 0.0;

            if (usage > 100.0)
                usage = 100.0;

            return static_cast<int>(usage + 0.5);
        }

        int SystemMonitor::getStorageUsage(const std::string& path) const
        {
            struct statvfs stat{};

            if (statvfs(path.c_str(), &stat) != 0)
                return -1;

            unsigned long long total =
                static_cast<unsigned long long>(stat.f_blocks) *
                stat.f_frsize;

            unsigned long long available =
                static_cast<unsigned long long>(stat.f_bavail) *
                stat.f_frsize;

            if (total == 0)
                return -1;

            unsigned long long used = total - available;

            double usage =
                static_cast<double>(used) /
                static_cast<double>(total) *
                100.0;

            return static_cast<int>(usage + 0.5);
        }

        int SystemMonitor::getMemoryUsage() const
        {
            std::ifstream file("/proc/meminfo");
            if (!file.is_open()) return -1;

            std::string key;
            unsigned long long value = 0;
            std::string unit;

            unsigned long long memTotal = 0;
            unsigned long long memAvailable = 0;

            while (file >> key >> value >> unit)
            {
                if (key == "MemTotal:")
                    memTotal = value;
                else if (key == "MemAvailable:")
                    memAvailable = value;

                if (memTotal > 0 && memAvailable > 0)
                    break;
            }

            if (memTotal == 0 || memAvailable > memTotal)
                return -1;

            double usage =
                static_cast<double>(memTotal - memAvailable) /
                static_cast<double>(memTotal) *
                100.0;

            return static_cast<int>(usage + 0.5);
        }
    }
}