#pragma once

#include <string>
#include <memory>
#include <functional>
#include <mqtt/async_client.h>
#include "fall-detection/vision/FallRuleEngine.hpp"

namespace fall_detection
{
    namespace network
    {
        /**
         * @brief MQTT 异步通信客户端类
         * 继承 virtual public mqtt::callback 以接收云端下发的消息
         */
        class MqttClient : public virtual mqtt::callback
        {
            public:
                // 定义回调函数类型，参数为（主题，消息内容）
                using MessageCallback = std::function<void(const std::string& topic, const std::string& payload)>;
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

                // 设置消息接收回调函数
                void setMessageCallback(MessageCallback callback);

                // 订阅指定主题
                bool subscribe(const std::string& topic, int qos = 1);

            protected:
                // 重写 Paho MQTT 的底层消息到达回调
                void message_arrived(mqtt::const_message_ptr msg) override;

                // 重写连接断开回调
                void connection_lost(const std::string& cause) override;
            
            private:
                std::string serverAddress_;
                std::string clientId_;
                int keepAliveSeconds_;

                // 使用 Paho MQTT 的现代 C++ 异步客户端，确保底层网络 I/O 
                std::unique_ptr<mqtt::async_client> client_;

                // 保存外部传入的回调函数
                MessageCallback messageCallback_;
        };
    }
}