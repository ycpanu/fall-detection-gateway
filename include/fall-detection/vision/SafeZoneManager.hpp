#pragma once

#include <cstddef>
#include <shared_mutex>
#include <string>
#include <vector>

namespace fall_detection
{
    namespace vision
    {
        /**
         * @brief 归一化二维坐标
         *
         * x、y 范围均为 [0, 1]。
         * 与实际摄像头分辨率无关。
         */
        struct NormalizedPoint
        {
            float x = 0.0f;
            float y = 0.0f;
        };


        /**
         * @brief 安全躺卧区域
         */
        struct SafeLieZone
        {
            std::string name;
            std::vector<NormalizedPoint> points;
        };


        class SafeZoneManager
        {
        public:
            SafeZoneManager() = default;

            /**
             * @brief 从 JSON 文件加载安全区域
             *
             * 启动时调用。
             */
            bool load(
                const std::string& configPath
            );


            /**
             * @brief 运行时替换安全区域并写入配置文件
             *
             * 后续 MQTT 配置下发时调用。
             *
             * 只有文件成功写入后，
             * 才会替换当前内存中的安全区域。
             */
            bool replaceZonesAndSave(
                const std::vector<SafeLieZone>& zones,
                const std::string& configPath
            );


            /**
             * @brief 获取当前安全区域副本
             *
             * 后续可用于：
             * - MQTT 查询
             * - Qt 配置回显
             * - 调试
             */
            std::vector<SafeLieZone> getZones() const;


            /**
             * @brief 获取区域数量
             */
            std::size_t getZoneCount() const;


            /**
             * @brief 判断一个归一化坐标点是否处于安全区域
             */
            bool isPointInSafeZone(
                float normalizedX,
                float normalizedY
            ) const;


            /**
             * @brief 计算人体框与安全区域的最大重叠比例
             */
            float calculateBoxOverlapRatio(
                int boxX,
                int boxY,
                int boxWidth,
                int boxHeight,
                int frameWidth,
                int frameHeight
            ) const;


        private:
            /**
             * @brief 点是否位于多边形内部
             */
            static bool isPointInPolygon(
                const NormalizedPoint& point,
                const std::vector<NormalizedPoint>& polygon
            );


            /**
             * @brief 校验外部下发的安全区域
             */
            static bool validateZone(
                const SafeLieZone& zone,
                std::string& errorMessage
            );


            /**
             * @brief 将完整区域配置写入 JSON
             */
            static bool writeConfigFile(
                const std::vector<SafeLieZone>& zones,
                const std::string& configPath
            );


        private:
            /*
             * FallRuleEngine 会频繁读取 zones_，
             * MQTT 配置线程只会偶尔修改。
             *
             * 因此使用 shared_mutex：
             *
             * 多个视觉读取线程：
             *      shared_lock
             *
             * 配置修改线程：
             *      unique_lock
             */
            mutable std::shared_mutex zonesMutex_;

            std::vector<SafeLieZone> zones_;
        };
    }
}