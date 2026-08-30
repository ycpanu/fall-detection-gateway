#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace fall_detection
{
    namespace utils
    {
        /**
         * @brief 全局配置管理器（单例）
         * 负责解析 configs/config.json，并以类型安全的方式向各模块分发运行参数。
         * 所有 getter 均内置默认值，即使配置文件缺失或字段不全，系统也能安全启动。
         */
        class ConfigManager
        {
            public:
                // 获取全局唯一实例
                static ConfigManager& getInstance();

                /**
                 * @brief 加载并解析 JSON 配置文件
                 * @param configPath 配置文件路径
                 * @return 是否加载成功
                 */
                bool load(const std::string& configPath);

                // 配置文件是否已成功加载
                bool isLoaded() const;

                // —— 通用取值接口（点号分层访问，如 "vision.camera_device_id"）——
                std::string getString(const std::string& key, const std::string& defaultValue = "") const;
                int getInt(const std::string& key, int defaultValue = 0) const;
                double getDouble(const std::string& key, double defaultValue = 0.0) const;
                bool getBool(const std::string& key, bool defaultValue = false) const;

                // —— 系统 ——
                std::string getLogFilePath() const;
                std::string getLogLevel() const;

                // —— 视觉采集 ——
                int getCameraDeviceId() const;
                int getFrameQueueSize() const;
                int getVideoCacheFrames() const;
                int getVideoSaveFps() const;
                std::string getVideoOutputDir() const;

                // —— 模型 ——
                std::string getRknnModelPath() const;
                float getConfidenceThreshold() const;
                float getNmsThreshold() const;

                // —— 规则引擎 ——
                float getKptConfThreshold() const;
                float getFallAngleThreshold() const;
                float getFallVelocityThreshold() const;
                int getConfirmFramesThreshold() const;
                int getStaticLieThreshold() const;
                int getFallEventWindow() const;

                // —— 网络 ——
                std::string getMqttBroker() const;
                std::string getMqttClientId() const;
                std::string getAlertTopic() const;
                int getKeepAliveSeconds() const;
                int getAlertQueueSize() const;

                // —— 硬件 ——
                int getBuzzerGpioPin() const;
                int getAlarmDurationMs() const;

                // —— 存储 ——
                std::string getSqliteDbPath() const;

            private:
                ConfigManager() = default;
                ~ConfigManager() = default;

                // 禁用拷贝，确保单例唯一
                ConfigManager(const ConfigManager&) = delete;
                ConfigManager& operator=(const ConfigManager&) = delete;

                // 按点号拆分 key 逐级下钻；未找到返回 nullptr
                const nlohmann::json* find(const std::string& key) const;

            private:
                nlohmann::json root_;
                bool loaded_ = false;
        };
    }
}
