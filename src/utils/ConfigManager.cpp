#include "fall-detection/utils/ConfigManager.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace fall_detection
{
    namespace utils
    {
        ConfigManager& ConfigManager::getInstance()
        {
            static ConfigManager instance;
            return instance;
        }

        bool ConfigManager::load(const std::string& configPath)
        {
            std::ifstream file(configPath);
            if (!file.is_open())
            {
                // 日志系统可能尚未初始化，这里直接打印到 stderr，避免依赖 spdlog
                std::cerr << "[ConfigManager] 无法打开配置文件：" << configPath << std::endl;
                loaded_ = false;
                return false;
            }

            try
            {
                file >> root_;
            }
            catch (const nlohmann::json::parse_error& e)
            {
                std::cerr << "[ConfigManager] 配置文件解析失败：" << e.what() << std::endl;
                loaded_ = false;
                return false;
            }

            loaded_ = true;
            return true;
        }

        bool ConfigManager::isLoaded() const
        {
            return loaded_;
        }

        const nlohmann::json* ConfigManager::find(const std::string& key) const
        {
            if (!loaded_)
            {
                return nullptr;
            }

            const nlohmann::json* node = &root_;
            std::stringstream ss(key);
            std::string part;
            while (std::getline(ss, part, '.'))
            {
                if (part.empty() || !node->is_object() || !node->contains(part))
                {
                    return nullptr;
                }
                node = &(*node)[part];
            }
            return node;
        }

        std::string ConfigManager::getString(const std::string& key, const std::string& defaultValue) const
        {
            const nlohmann::json* node = find(key);
            return (node != nullptr && node->is_string()) ? node->get<std::string>() : defaultValue;
        }

        int ConfigManager::getInt(const std::string& key, int defaultValue) const
        {
            const nlohmann::json* node = find(key);
            return (node != nullptr && node->is_number_integer()) ? node->get<int>() : defaultValue;
        }

        double ConfigManager::getDouble(const std::string& key, double defaultValue) const
        {
            const nlohmann::json* node = find(key);
            return (node != nullptr && node->is_number()) ? node->get<double>() : defaultValue;
        }

        bool ConfigManager::getBool(const std::string& key, bool defaultValue) const
        {
            const nlohmann::json* node = find(key);
            return (node != nullptr && node->is_boolean()) ? node->get<bool>() : defaultValue;
        }

        // —— 系统 ——
        std::string ConfigManager::getLogFilePath() const { return getString("system.log_file_path", "logs/gateway.log"); }
        std::string ConfigManager::getLogLevel() const     { return getString("system.log_level", "INFO"); }

        // —— 视觉采集 ——
        int ConfigManager::getCameraDeviceId() const   { return getInt("vision.camera_device_id", 0); }
        int ConfigManager::getFrameQueueSize() const   { return getInt("vision.frame_queue_size", 3); }
        int ConfigManager::getVideoCacheFrames() const { return getInt("vision.video_cache_frames", 90); }
        int ConfigManager::getVideoSaveFps() const     { return getInt("vision.video_save_fps", 30); }
        std::string ConfigManager::getVideoOutputDir() const   { return getString("vision.video_output_dir", "videos"); }

        // —— 模型 ——
        std::string ConfigManager::getRknnModelPath() const       { return getString("model.rknn_model_path", "./yolov8n-pose.rknn"); }
        float ConfigManager::getConfidenceThreshold() const { return static_cast<float>(getDouble("model.confidence_threshold", 0.5)); }
        float ConfigManager::getNmsThreshold() const        { return static_cast<float>(getDouble("model.nms_threshold", 0.45)); }

        // —— 规则引擎 ——
        float ConfigManager::getKptConfThreshold() const       { return static_cast<float>(getDouble("rule_engine.kpt_conf_threshold", 0.3)); }
        float ConfigManager::getFallAngleThreshold() const     { return static_cast<float>(getDouble("rule_engine.fall_angle_threshold", 60.0)); }
        float ConfigManager::getFallVelocityThreshold() const  { return static_cast<float>(getDouble("rule_engine.fall_velocity_threshold", 400.0)); }
        int ConfigManager::getConfirmFramesThreshold() const { return getInt("rule_engine.confirm_frames_threshold", 5); }
        int ConfigManager::getStaticLieThreshold() const     { return getInt("rule_engine.static_lie_threshold", 30); }
        int ConfigManager::getFallEventWindow() const        { return getInt("rule_engine.fall_event_window", 15); }

        // —— 网络 ——
        std::string ConfigManager::getMqttBroker() const       { return getString("network.mqtt_broker", "tcp://broker.emqx.io:1883"); }
        std::string ConfigManager::getMqttClientId() const     { return getString("network.mqtt_client_id", "OrangePi_Gateway_001"); }
        std::string ConfigManager::getAlertTopic() const       { return getString("network.alert_topic", "fall_detection/alerts"); }
        int ConfigManager::getKeepAliveSeconds() const { return getInt("network.keep_alive_seconds", 20); }
        int ConfigManager::getAlertQueueSize() const   { return getInt("network.alert_queue_size", 10); }

        // —— 硬件 ——
        int ConfigManager::getBuzzerGpioPin() const   { return getInt("hardware.buzzer_gpio_pin", 138); }
        int ConfigManager::getAlarmDurationMs() const { return getInt("hardware.alarm_duration_ms", 3000); }

        // —— 存储 ——
        std::string ConfigManager::getSqliteDbPath() const { return getString("storage.sqlite_db_path", "fall_detection.db"); }
    }
}
