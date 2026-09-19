#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace fall_detection
{
    namespace vision
    {
        /**
         * @brief 归一化坐标点
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

        /**
         * @brief 场景安全区域管理器
         */
        class SafeZoneManager
        {
        public:
            SafeZoneManager() = default;

            /**
             * @brief 加载场景配置文件
             */
            bool load(const std::string& configPath);

            /**
             * @brief 判断归一化坐标是否位于安全区域
             */
            bool isPointInSafeZone(float normalizedX, float normalizedY) const;

            /**
             * @brief 当前安全区域数量
             */
            std::size_t getZoneCount() const;

            /**
             * @brief 计算人体框与安全区域中最大重叠比例
             */
            float calculateBoxOverlapRatio(int boxX, int boxY, int boxWidth, int boxHeight, int frameWidth, int frameHeight) const;

        private:
            /**
             * @brief 判断点是否位于指定多边形内部
             * 使用射线法
             */
            bool isPointInPolygon(const NormalizedPoint& point, const std::vector<NormalizedPoint>& polygon) const;

        private:
            std::vector<SafeLieZone> zones_;

        };
    }
}