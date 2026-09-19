#include "fall-detection/vision/SafeZoneManager.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>


namespace fall_detection
{
    namespace vision
    {
        bool SafeZoneManager::load(const std::string& configPath)
        {
            zones_.clear();

            std::ifstream file(configPath);

            if (!file.is_open())
            {
                LOG_WARN(
                    "[SafeZone] 无法打开场景配置文件：{}，"
                    "当前不启用安全躺卧区域",
                    configPath
                );

                return false;
            }


            try
            {
                nlohmann::json root;
                file >> root;


                if (!root.contains("safe_lie_zones") ||
                    !root["safe_lie_zones"].is_array())
                {
                    LOG_WARN(
                        "[SafeZone] scene_config.json "
                        "中不存在有效 safe_lie_zones"
                    );

                    return false;
                }


                for (const auto& zoneJson :
                     root["safe_lie_zones"])
                {
                    if (!zoneJson.contains("name") ||
                        !zoneJson.contains("points"))
                    {
                        continue;
                    }


                    SafeLieZone zone;

                    zone.name =
                        zoneJson.value(
                            "name",
                            "unnamed_zone"
                        );


                    const auto& pointsJson =
                        zoneJson["points"];


                    if (!pointsJson.is_array())
                    {
                        continue;
                    }


                    for (const auto& pointJson :
                         pointsJson)
                    {
                        if (!pointJson.is_array() ||
                            pointJson.size() != 2)
                        {
                            continue;
                        }


                        NormalizedPoint point;

                        point.x =
                            pointJson[0].get<float>();

                        point.y =
                            pointJson[1].get<float>();


                        /*
                         * 防止错误配置出现负坐标
                         * 或者超过 1.0。
                         */
                        if (point.x < 0.0f ||
                            point.x > 1.0f ||
                            point.y < 0.0f ||
                            point.y > 1.0f)
                        {
                            LOG_WARN(
                                "[SafeZone] 区域 {} "
                                "包含非法坐标 ({}, {})",
                                zone.name,
                                point.x,
                                point.y
                            );

                            continue;
                        }


                        zone.points.push_back(point);
                    }


                    /*
                     * 多边形至少需要三个点。
                     */
                    if (zone.points.size() < 3)
                    {
                        LOG_WARN(
                            "[SafeZone] 区域 {} 点数不足，已忽略",
                            zone.name
                        );

                        continue;
                    }


                    zones_.push_back(zone);

                    LOG_INFO(
                        "[SafeZone] 已加载安全区域：{}，{} 个顶点",
                        zone.name,
                        zone.points.size()
                    );
                }


                LOG_INFO(
                    "[SafeZone] 共加载 {} 个安全躺卧区域",
                    zones_.size()
                );

                return true;
            }
            catch (const std::exception& e)
            {
                LOG_ERROR(
                    "[SafeZone] 解析场景配置失败：{}",
                    e.what()
                );

                zones_.clear();

                return false;
            }
        }


        bool SafeZoneManager::isPointInSafeZone(
            float normalizedX,
            float normalizedY) const
        {
            NormalizedPoint point;

            point.x = normalizedX;
            point.y = normalizedY;


            for (const auto& zone : zones_)
            {
                if (isPointInPolygon(
                        point,
                        zone.points))
                {
                    return true;
                }
            }
            return false;
        }


        std::size_t SafeZoneManager::getZoneCount() const
        {
            return zones_.size();
        }


        bool SafeZoneManager::isPointInPolygon(
            const NormalizedPoint& point,
            const std::vector<NormalizedPoint>& polygon) const
        {
            if (polygon.size() < 3)
            {
                return false;
            }


            bool inside = false;

            std::size_t j = polygon.size() - 1;


            /*
             * 射线法：
             *
             * 从待判断点向右画一条水平射线。
             * 如果与多边形边的交点数量为奇数，
             * 则点位于多边形内部。
             */
            for (std::size_t i = 0;
                 i < polygon.size();
                 ++i)
            {
                const auto& pi = polygon[i];
                const auto& pj = polygon[j];


                const bool intersect =
                    ((pi.y > point.y) !=
                     (pj.y > point.y))
                    &&
                    (
                        point.x <
                        (pj.x - pi.x)
                        *
                        (point.y - pi.y)
                        /
                        (pj.y - pi.y + 1e-6f)
                        +
                        pi.x
                    );


                if (intersect)
                {
                    inside = !inside;
                }


                j = i;
            }


            return inside;
        }

        float SafeZoneManager::calculateBoxOverlapRatio(
            int boxX,
            int boxY,
            int boxWidth,
            int boxHeight,
            int frameWidth,
            int frameHeight) const
        {
            // 基本参数检查
            if (zones_.empty() ||
                boxWidth <= 0 ||
                boxHeight <= 0 ||
                frameWidth <= 0 ||
                frameHeight <= 0)
            {
                return 0.0f;
            }


            /*
            * 将人体框限制在实际画面范围内。
            *
            * YOLO 输出的人体框理论上应该位于画面中，
            * 但边界情况下仍然做一次保护。
            */
            const int left =
                std::clamp(boxX, 0, frameWidth);

            const int top =
                std::clamp(boxY, 0, frameHeight);

            const int right =
                std::clamp(
                    boxX + boxWidth,
                    0,
                    frameWidth
                );

            const int bottom =
                std::clamp(
                    boxY + boxHeight,
                    0,
                    frameHeight
                );


            const int clippedWidth =
                right - left;

            const int clippedHeight =
                bottom - top;


            if (clippedWidth <= 0 ||
                clippedHeight <= 0)
            {
                return 0.0f;
            }


            /*
            * 只创建人体框大小的 Mask，
            * 而不是创建整张摄像头画面的 Mask。
            *
            * 这样计算量比较小。
            */
            cv::Mat mask =
                cv::Mat::zeros(
                    clippedHeight,
                    clippedWidth,
                    CV_8UC1
                );


            float maxOverlapRatio = 0.0f;


            for (const auto& zone : zones_)
            {
                if (zone.points.size() < 3)
                {
                    continue;
                }


                std::vector<cv::Point> polygon;

                polygon.reserve(
                    zone.points.size()
                );


                /*
                * scene_config.json 中保存的是归一化坐标。
                *
                * 先恢复成摄像头实际像素坐标，
                * 再转换成人体框局部坐标。
                */
                for (const auto& point :
                    zone.points)
                {
                    const int pixelX =
                        static_cast<int>(
                            point.x *
                            static_cast<float>(frameWidth)
                        );

                    const int pixelY =
                        static_cast<int>(
                            point.y *
                            static_cast<float>(frameHeight)
                        );


                    /*
                    * mask 的 (0,0)
                    * 实际对应原始画面中的 (left, top)。
                    */
                    polygon.emplace_back(
                        pixelX - left,
                        pixelY - top
                    );
                }


                // 每个安全区域单独计算
                cv::Mat zoneMask =
                    cv::Mat::zeros(
                        clippedHeight,
                        clippedWidth,
                        CV_8UC1
                    );


                std::vector<
                    std::vector<cv::Point>
                > polygons;

                polygons.push_back(
                    std::move(polygon)
                );


                /*
                * 在人体框局部 Mask 上绘制安全区域。
                *
                * 超出人体框的部分会自动被裁剪，
                * 因此留下来的白色部分就是：
                *
                * 人体框 ∩ 安全区域
                */
                cv::fillPoly(
                    zoneMask,
                    polygons,
                    cv::Scalar(255)
                );


                const int intersectionArea =
                    cv::countNonZero(zoneMask);


                const int boxArea =
                    clippedWidth *
                    clippedHeight;


                if (boxArea <= 0)
                {
                    continue;
                }


                const float overlapRatio =
                    static_cast<float>(
                        intersectionArea
                    )
                    /
                    static_cast<float>(
                        boxArea
                    );


                maxOverlapRatio =
                    std::max(
                        maxOverlapRatio,
                        overlapRatio
                    );
            }


            return maxOverlapRatio;
        }
    }
}