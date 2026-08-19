#pragma once

#include <string>
#include <memory>
#include <mqtt/async_client.h>
#include "fall-detection/vision/FallRuleEngine.hpp"

namespace fall_detection
{
    namespace network
    {
        /**
         * @brief MQTT 异步通信客户端类
         * 负责维护与云端物联网平台的长连接，当确认摔倒时，将报警事件序列化为 JSON 格式，并以 Qos 1（至少到达一次）的级别发布到云端
         */
        class MqttClient
        {
            public:
                /**
                 * @brief 构造函数
                 * @param serverAddress MQTT 服务器/Broker 的地址 (例如 "tcp://broker.emqx.io:1883")
                 * @param clientId 边缘网关的唯一设备标识符
                 * @param keepAliveSeconds 心跳保活间隔（秒），弱网环境不宜过大
                 */
                MqttClient(const std::string& serverAddress, const std::string& clientId, int keepAliveSeconds = 20);

                ~MqttClient();

                /**
                 * @brief 连接到 MQTT 云端服务器
                 * @return 是否连接成功
                 */
                bool connect();

                /**
                 * @brief 断开与服务器的连接
                 * 
                 */
                void disconnect();

                /**
                 * @brief 发布摔倒报警事件
                 * @param topic 发布的主题 (例如 "gateway/fall_alert")
                 * @param event 规则引擎产生的报警事件数据
                 * @return 消息是否成功放入发送队列
                 */
                bool publishAlert(const std::string& topic, const vision::AlertEvent& event);

                /**
                 * @brief 查询与 MQTT 服务器的连接状态
                 * @return 是否已连接
                 */
                bool isConnected() const;
            
            private:
                std::string serverAddress_;
                std::string clientId_;
                int keepAliveSeconds_;

                // 使用 Paho MQTT 的现代 C++ 异步客户端，确保底层网络 I/O 
                std::unique_ptr<mqtt::async_client> client_;
        };
    }
}