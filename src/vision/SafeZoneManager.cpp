#include "fall-detection/vision/SafeZoneManager.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <system_error>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>


namespace fall_detection
{
    namespace vision
    {
        bool SafeZoneManager::load(const std::string& configPath)
        {
            std::ifstream file(configPath);

            if (!file.is_open())
            {
                LOG_WARN(
                    "[SafeZone] 无法打开场景配置文件：{}，"
                    "保留当前安全区域配置",
                    configPath
                );

                return false;
            }


            std::vector<SafeLieZone> newZones;

            try
            {
                nlohmann::json root;

                file >> root;

                if (!root.contains("safe_lie_zones") || !root["safe_lie_zones"].is_array())
                {
                    LOG_WARN(
                        "[SafeZone] 配置文件中不存在有效 "
                        "safe_lie_zones"
                    );

                    return false;
                }


                for (const auto& zoneJson : root["safe_lie_zones"])
                {
                    /*
                     * 保持旧版本行为：
                     * 某个区域配置错误时跳过，
                     * 不影响其他合法区域。
                     */
                    if (!zoneJson.is_object() || !zoneJson.contains("points"))
                    {
                        continue;
                    }

                    SafeLieZone zone;

                    zone.name = zoneJson.value(
                            "name",
                            "unnamed_zone"
                        );

                    const auto& pointsJson = zoneJson["points"];

                    if (!pointsJson.is_array())
                    {
                        continue;
                    }


                    for (const auto& pointJson : pointsJson)
                    {
                        if (!pointJson.is_array() ||
                            pointJson.size() != 2)
                        {
                            continue;
                        }

                        try
                        {
                            const float x = pointJson[0].get<float>();

                            const float y = pointJson[1].get<float>();


                            if (!std::isfinite(x) ||
                                !std::isfinite(y) ||
                                x < 0.0f ||
                                x > 1.0f ||
                                y < 0.0f ||
                                y > 1.0f)
                            {
                                LOG_WARN(
                                    "[SafeZone] 区域 {} "
                                    "包含非法坐标 ({}, {})",
                                    zone.name,
                                    x,
                                    y
                                );

                                continue;
                            }


                            zone.points.push_back( NormalizedPoint{x, y});
                        }
                        catch (const std::exception&)
                        {
                            LOG_WARN(
                                "[SafeZone] 区域 {} "
                                "存在无法解析的坐标",
                                zone.name
                            );
                        }
                    }


                    if (zone.points.size() < 3)
                    {
                        LOG_WARN(
                            "[SafeZone] 区域 {} 点数不足，已忽略",
                            zone.name
                        );

                        continue;
                    }


                    newZones.push_back(std::move(zone));
                }


                /*
                 * 全部解析成功以后，
                 * 再一次性替换内存数据。
                 *
                 * 避免 load() 失败时把原来的
                 * 正常安全区域清掉。
                 */
                {
                    std::unique_lock<std::shared_mutex>
                        lock(zonesMutex_);

                    zones_ = std::move(newZones);
                }


                LOG_INFO(
                    "[SafeZone] 配置加载完成，共 {} 个安全区域",
                    getZoneCount()
                );


                return true;
            }
            catch (const std::exception& e)
            {
                LOG_ERROR(
                    "[SafeZone] 解析场景配置失败：{}",
                    e.what()
                );

                /*
                 * 注意：
                 *
                 * 这里不清空 zones_。
                 *
                 * 如果运行过程中重新加载失败，
                 * 应继续使用上一份有效配置。
                 */
                return false;
            }
        }


        bool SafeZoneManager::replaceZonesAndSave(
            const std::vector<SafeLieZone>& zones,
            const std::string& configPath)
        {
            /*
             * 1. 先完整校验所有外部配置。
             */
            for (const auto& zone : zones)
            {
                std::string errorMessage;

                if (!validateZone(
                        zone,
                        errorMessage))
                {
                    LOG_ERROR(
                        "[SafeZone] 安全区域配置非法："
                        "name={}, error={}",
                        zone.name,
                        errorMessage
                    );

                    return false;
                }
            }


            /*
             * 2. 先落盘。
             *
             * 只有配置文件写成功，
             * 才更新运行时内存。
             *
             * 这样可以保证：
             *
             * 内存配置
             *     和
             * scene_config.json
             *
             * 不会出现一边成功、一边失败。
             */
            if (!writeConfigFile(zones, configPath))
            {
                return false;
            }

            /*
             * 3. 更新运行时配置。
             */
            {
                std::unique_lock<std::shared_mutex>lock(zonesMutex_);

                zones_ = zones;
            }


            LOG_INFO(
                "[SafeZone] 安全区域已热更新，共 {} 个区域",
                zones.size()
            );


            return true;
        }


        std::vector<SafeLieZone>
        SafeZoneManager::getZones() const
        {
            std::shared_lock<std::shared_mutex> lock(zonesMutex_);

            return zones_;
        }


        std::size_t
        SafeZoneManager::getZoneCount() const
        {
            std::shared_lock<std::shared_mutex>
                lock(zonesMutex_);

            return zones_.size();
        }


        bool SafeZoneManager::isPointInSafeZone(
            float normalizedX,
            float normalizedY) const
        {
            if (!std::isfinite(normalizedX) ||
                !std::isfinite(normalizedY))
            {
                return false;
            }


            const NormalizedPoint point{
                normalizedX,
                normalizedY
            };


            /*
             * 视觉线程只读取，
             * 使用共享读锁。
             */
            std::shared_lock<std::shared_mutex> lock(zonesMutex_);


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


        bool SafeZoneManager::isPointInPolygon(
            const NormalizedPoint& point,
            const std::vector<NormalizedPoint>& polygon)
        {
            if (polygon.size() < 3)
            {
                return false;
            }


            bool inside = false;

            std::size_t j = polygon.size() - 1;

            /*
             * 射线法判断点是否位于多边形内部。
             */
            for (std::size_t i = 0; i < polygon.size(); ++i)
            {
                const auto& pi = polygon[i];

                const auto& pj = polygon[j];


                const bool intersect =
                    ((pi.y > point.y) !=
                     (pj.y > point.y))
                    &&
                    (
                        point.x <
                        (pj.x - pi.x) *
                        (point.y - pi.y) /
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


        bool SafeZoneManager::validateZone(const SafeLieZone& zone,std::string& errorMessage)
        {
            if (zone.name.empty())
            {
                errorMessage =
                    "区域名称不能为空";

                return false;
            }


            if (zone.points.size() < 3)
            {
                errorMessage =
                    "多边形至少需要 3 个顶点";

                return false;
            }


            for (std::size_t i = 0; i < zone.points.size(); ++i)
            {
                const auto& point = zone.points[i];

                if (!std::isfinite(point.x) || !std::isfinite(point.y))
                {
                    errorMessage = "坐标包含非有限数值";

                    return false;
                }


                if (point.x < 0.0f ||
                    point.x > 1.0f ||
                    point.y < 0.0f ||
                    point.y > 1.0f)
                {
                    errorMessage =
                        "第 " +
                        std::to_string(i + 1) +
                        " 个坐标不在 [0, 1] 范围内";

                    return false;
                }
            }


            return true;
        }


        bool SafeZoneManager::writeConfigFile(
            const std::vector<SafeLieZone>& zones,
            const std::string& configPath)
        {
            try
            {
                nlohmann::json root;

                root["version"] = 1;

                root["safe_lie_zones"] = nlohmann::json::array();


                for (const auto& zone : zones)
                {
                    nlohmann::json zoneJson;

                    zoneJson["name"] = zone.name;

                    zoneJson["points"] = nlohmann::json::array();

                    for (const auto& point : zone.points)
                    {
                        zoneJson["points"].push_back(
                            {
                                point.x,
                                point.y
                            }
                        );
                    }


                    root["safe_lie_zones"].push_back(std::move(zoneJson));
                }

                const std::filesystem::path targetPath(configPath);

                /*
                 * 确保配置目录存在。
                 */
                if (targetPath.has_parent_path())
                {
                    std::filesystem::create_directories(
                        targetPath.parent_path()
                    );
                }

                /*
                 * 不直接覆盖正式配置。
                 *
                 * 先写 .tmp，
                 * 成功后 rename。
                 *
                 * 避免程序异常退出时产生半份 JSON。
                 */
                const std::filesystem::path tempPath = targetPath.string() + ".tmp";

                {
                    std::ofstream output(tempPath,std::ios::out |std::ios::trunc);

                    if (!output.is_open())
                    {
                        LOG_ERROR(
                            "[SafeZone] 无法创建临时配置文件：{}",
                            tempPath.string()
                        );

                        return false;
                    }


                    output << root.dump(2) << '\n';

                    output.flush();

                    if (!output.good())
                    {
                        LOG_ERROR(
                            "[SafeZone] 写入临时配置文件失败：{}",
                            tempPath.string()
                        );

                        return false;
                    }
                }


                std::error_code ec;


                /*
                 * Orange Pi 运行 Linux，
                 * rename 在同一文件系统内完成替换，
                 * 避免正式配置出现半写状态。
                 */
                std::filesystem::rename(
                    tempPath,
                    targetPath,
                    ec
                );


                if (ec)
                {
                    LOG_ERROR(
                        "[SafeZone] 替换配置文件失败：{}",
                        ec.message()
                    );


                    std::error_code removeEc;

                    std::filesystem::remove(
                        tempPath,
                        removeEc
                    );


                    return false;
                }


                LOG_INFO(
                    "[SafeZone] 配置已保存：{}",
                    configPath
                );


                return true;
            }
            catch (const std::exception& e)
            {
                LOG_ERROR(
                    "[SafeZone] 保存配置失败：{}",
                    e.what()
                );

                return false;
            }
        }


        float SafeZoneManager::calculateBoxOverlapRatio(
            int boxX,
            int boxY,
            int boxWidth,
            int boxHeight,
            int frameWidth,
            int frameHeight) const
        {
            if (boxWidth <= 0 ||
                boxHeight <= 0 ||
                frameWidth <= 0 ||
                frameHeight <= 0)
            {
                return 0.0f;
            }


            /*
             * 整个重叠计算期间保护 zones_。
             */
            std::shared_lock<std::shared_mutex>
                lock(zonesMutex_);


            if (zones_.empty())
            {
                return 0.0f;
            }


            const int left =
                std::clamp(
                    boxX,
                    0,
                    frameWidth
                );


            const int top =
                std::clamp(
                    boxY,
                    0,
                    frameHeight
                );


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


            const int clippedWidth = right - left;

            const int clippedHeight = bottom - top;

            if (clippedWidth <= 0 || clippedHeight <= 0)
            {
                return 0.0f;
            }

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


                for (const auto& point : zone.points)
                {
                    const int pixelX =
                        static_cast<int>(
                            point.x *
                            static_cast<float>(
                                frameWidth
                            )
                        );


                    const int pixelY =
                        static_cast<int>(
                            point.y *
                            static_cast<float>(
                                frameHeight
                            )
                        );


                    polygon.emplace_back(
                        pixelX - left,
                        pixelY - top
                    );
                }


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


                cv::fillPoly(
                    zoneMask,
                    polygons,
                    cv::Scalar(255)
                );


                const int intersectionArea =
                    cv::countNonZero(
                        zoneMask
                    );


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