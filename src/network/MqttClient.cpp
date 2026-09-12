#include "fall-detection/network/MqttClient.hpp"
#include "fall-detection/utils/SysLogger.hpp"
#include <nlohmann/json.hpp>
#include <chrono>

using json = nlohmann::json;

namespace fall_detection
{
    namespace network
    {
        MqttClient::MqttClient(const std::string& serverAddress, const std::string& clientId, int keepAliveSeconds)
            : serverAddress_(serverAddress)
            , clientId_(clientId)
            , keepAliveSeconds_(keepAliveSeconds)
        {
            // 实例化 Paho MQTT 异步客户端
            client_ = std::make_unique<mqtt::async_client>(serverAddress_, clientId_);

            // 将当前类注册为 MQTT 客户端的回调处理对象
            client_->set_callback(*this);
        }

        MqttClient::~MqttClient()
        {
            disconnect();
        }

        bool MqttClient::connect()
        {
            if (isConnected())
            {
                LOG_WARN("MQTT 客户端已连接！");
                return true;
            }

            LOG_INFO("准备连接 MQTT 服务器：{}, 设备ID：{}", serverAddress_, clientId_);

            try
            {
                // 配置连接选项
                mqtt::connect_options connOpts;
                connOpts.set_clean_session(true);
                connOpts.set_keep_alive_interval(keepAliveSeconds_);   // 心跳保活间隔，适应弱网环境

                // 断网自动重连
                connOpts.set_automatic_reconnect(1, 10);

                // 阻塞等待首次连接结果
                mqtt::token_ptr conntok = client_->connect(connOpts);
                conntok->wait_for(std::chrono::seconds(5));

                LOG_INFO("MQTT 服务器连接成功！");
                return true;
            }
            catch (const mqtt::exception& exc)
            {
                LOG_ERROR("MQTT 连接失败！错误信息：{}", exc.what());
                return false;
            }
        }

        void MqttClient::disconnect()
        {
            if (isConnected())
            {
                try
                {
                    LOG_INFO("正在断开 MQTT 连接...");
                    client_->disconnect()->wait();
                    LOG_INFO("MQTT 连接已安全断开！");
                }
                catch(const mqtt::exception& exc)
                {
                    LOG_ERROR("MQTT 断开连接时发生异常：{}", exc.what());
                }
                
            }
        }

        bool MqttClient::isConnected() const
        {
            return client_ != nullptr && client_->is_connected();
        }

        bool MqttClient::publishAlert(const std::string& topic, const vision::AlertEvent& event)
        {
            // 探针检测
            if (!isConnected())
            {
                LOG_ERROR("发送警报失败：MQTT 处于离线状态！");
                return false;
            }

            try 
            {
                // 1. 利用 nlohmann/json 将 C++ 结构体数据装为标准 JSON 格式
                json payloadJson;
                payloadJson["device_id"] = clientId_;
                payloadJson["event_type"] = "FALL_DETECTED";
                payloadJson["timestamp"] = event.timestamp;
                payloadJson["data"]["trigger_x"] = event.triggerBoxX;
                payloadJson["data"]["trigger_y"] = event.triggerBoxY;
                payloadJson["data"]["status"] = "CRITICAL";

                // 2. 序列化字符串
                std::string payloadStr = payloadJson.dump();

                // 3. 构建 QMTT 消息（QoS 1)
                mqtt::message_ptr pubmsg = mqtt::make_message(topic, payloadStr);
                pubmsg->set_qos(1);
                pubmsg->set_retained(false);

                // 4. 异步发布
                client_->publish(pubmsg);

                LOG_INFO("成功向 {} 发布报警数据：{}", topic, payloadStr);
                return true;
            }
            catch (const mqtt::exception& exc)
            {
                LOG_ERROR("MQTT 消息发布异常：{}", exc.what());
                return false;
            }
        }

        // 保存外部传入的 Lambda 表达式
        void MqttClient::setMessageCallback(MessageCallback cb)
        {
            messageCallback_ = cb;
        }

        // 向云端发送订阅请求
        bool MqttClient::subscribe(const std::string& topic, int qos)
        {
            if (!isConnected())
            {
                LOG_ERROR("订阅主题失败：MQTT 处于离线状态！");
                return false;
            }

            try
            {
                client_->subscribe(topic, qos)->wait();
                LOG_INFO("成功订阅主题：{} (QoS {})", topic, qos);
                return true;
            }
            catch (const mqtt::exception& exc)
            {
                LOG_ERROR("MQTT 订阅主题异常：{}", exc.what());
                return false;
            }
        }

        // 底层收到消息时自动触发此函数
        void MqttClient::message_arrived(mqtt::const_message_ptr msg)
        {
            std::string topic = msg->get_topic();
            std::string payload = msg->to_string();

            LOG_INFO("收到云端指令：主题 [{}]，内容 [{}]", topic, payload);

            // 如果 main.cpp 里设置了回调函数，则把数据传过去
            if (messageCallback_)
            {
                messageCallback_(topic, payload);
            }
        }

        // 连接意外断开时触发此函数
        void MqttClient::connection_lost(const std::string& cause)
        {
            LOG_WARN("MQTT 连接意外断开，原因：{}", cause);
        }
    }
}