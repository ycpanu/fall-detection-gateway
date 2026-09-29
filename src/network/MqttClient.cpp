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
                conntok->wait();

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

        bool MqttClient::publishAlert(const std::string& topic, const fall_detection::event::AlertEvent& alertEvent)
        {
            if (!isConnected())
            {
                LOG_ERROR("发送报警失败：MQTT处于离线状态");

                return false;
            }

            try
            {
                json payloadJson;

                // 事件基本信息
                payloadJson["event_id"] = alertEvent.eventId;

                payloadJson["device_id"] = alertEvent.deviceId.empty() ? clientId_ : alertEvent.deviceId;

                payloadJson["deployment_area"] = alertEvent.deploymentArea;

                payloadJson["event_type"] = fall_detection::event::toString(alertEvent.eventType);

                payloadJson["timestamp"] = alertEvent.timestamp;

                payloadJson["status"] = fall_detection::event::toString(alertEvent.status);

                // 事件来源
                payloadJson["sources"] = json::array();

                for (const auto source : alertEvent.source)
                {
                    payloadJson["sources"].push_back(fall_detection::event::toString(source));
                }

                // 事件附加信息
                payloadJson["data"]["person_track_id"] = alertEvent.personTrackId;

                payloadJson["data"]["trigger_x"] = alertEvent.triggerBoxX;

                payloadJson["data"]["trigger_y"] = alertEvent.triggerBoxY;


                if (!alertEvent.keyword.empty())
                {
                    payloadJson["data"]["keyword"] = alertEvent.keyword;
                }

                /*
                * 报警视频后续通过 HTTP 单独上传。
                * 当前 MQTT 不发送本地文件内容。
                */
                payloadJson["data"]["video_url"] = "";

                const std::string payloadStr = payloadJson.dump();

                mqtt::message_ptr pubmsg = mqtt::make_message(topic, payloadStr);

                pubmsg->set_qos(1);
                pubmsg->set_retained(false);

                // QoS 1 发布，等待 Broker 的 PUBACK
                auto token = client_->publish(pubmsg);

                constexpr auto ACK_TIMEOUT = std::chrono::seconds(5);

                if (!token->wait_for(ACK_TIMEOUT))
                {
                    LOG_ERROR(
                        "MQTT 报警发送超时，未收到 Broker 确认：event_id={}",
                        alertEvent.eventId
                    );

                    return false;
                }

                LOG_INFO(
                    "MQTT 报警已收到 Broker QoS1 确认：event_id={}",
                    alertEvent.eventId
                );

                return true;
            }
            catch (const mqtt::exception& exc)
            {
                LOG_ERROR("MQTT报警消息发布异常：{}", exc.what());

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
            // 1. 保存订阅信息
            //    后续 MQTT 自动重连以后需要重新订阅
            {
                std::lock_guard<std::mutex> lock(subscriptionMtx_);

                bool exists = false;

                for (auto& item : subscriptions_)
                {
                    if (item.first == topic)
                    {
                        item.second = qos;
                        exists = true;
                        break;
                    }
                }

                if (!exists)
                {
                    subscriptions_.emplace_back(topic, qos);
                }
            }

            // 2. 当前离线时只记录订阅
            if (!isConnected())
            {
                LOG_WARN("MQTT 当前离线，已记录待恢复订阅：{}", topic);

                return false;
            }

            // 3. 当前在线，立即订阅
            try
            {
                client_->subscribe(topic, qos)->wait();

                LOG_INFO("MQTT 主题订阅成功：topic={}, qos={}",topic, qos);

                return true;
            }
            catch (const mqtt::exception& exc)
            {
                LOG_ERROR("MQTT 主题订阅失败：topic={}, error={}",topic,exc.what());

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

        void MqttClient::connected(const std::string& cause)
        {
            if (cause.empty())
            {
                LOG_INFO("MQTT 连接建立成功");
            }
            else
            {
                LOG_INFO("MQTT 自动重连成功: {}", cause);
            }

            std::vector<std::pair<std::string, int>> subscriptionsCopy;

            {
                std::lock_guard<std::mutex> lock(subscriptionMtx_);
                subscriptionsCopy = subscriptions_;
            }

            // connected() 运行在 Paho MQTT 的内部回调线程中，这里不能调用 token->wait()
            // 否则会阻塞 Paho 自己的网络处理线程，引发死锁
            for (const auto& item : subscriptionsCopy)
            {
                try 
                {
                    client_->subscribe(item.first, item.second);
                    LOG_INFO("MQTT 已提交自动恢复订阅: topic={}, qos={}", item.first, item.second);
                }
                catch(const mqtt::exception& exc)
                {
                    LOG_ERROR("MQTT 恢复订阅失败: topic={}, error={}", item.first, exc.what());
                }
            }
        }

        // 用于发送低负载的心跳包
        bool MqttClient::publishStatus(const std::string& topic, int cpuUsage, int memoryUsage,
            int storageUsage, const std::string& deploymentArea)
        {
            if (!isConnected()) return false;

            try
            {
                json payloadJson;

                payloadJson["device_id"] = clientId_;
                payloadJson["deployment_area"] = deploymentArea;
                payloadJson["cpu_usage"] = cpuUsage;
                payloadJson["memory_usage"] = memoryUsage;
                payloadJson["storage_usage"] = storageUsage;

                payloadJson["timestamp"] =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();

                mqtt::message_ptr pubmsg =
                    mqtt::make_message(topic, payloadJson.dump());

                pubmsg->set_qos(0);
                pubmsg->set_retained(false);

                client_->publish(pubmsg);

                return true;
            }
            catch (const mqtt::exception& exc)
            {
                LOG_ERROR("设备状态上报失败：{}", exc.what());
                return false;
            }
        }
    }
}