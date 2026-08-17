#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/concurrency/ThreadSafeQueue.hpp"
#include "fall-detection/vision/CameraStreamer.hpp"
#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/network/MqttClient.hpp"
#include "fall-detection/hardware/BuzzerController.hpp"
#include "fall-detection/utils/LocalDatabase.hpp"

using namespace fall_detection;

int main(int argc, char** argv)
{
    // 初始化全局日志系统
    utils::SysLogger::getInstance().init("logs/gateway.log");
    LOG_INFO("系统启动！");

    // 初始化底层硬件与容灾模块
    hardware::BuzzerController buzzer(73);
    buzzer.init();

    utils::LocalDatabase db("fall_detection.db");
    db.init();
    
    // 初始化网络通信层
    network::MqttClient mqttClient("tcp://broker.emqx.io:1883", "Orangepi_Gateway_001");
    mqttClient.connect();

    // 初始化 NPU 硬件推理
    RKNNInferencer inferencer("./best.rknn");
    if (!inferencer.init())
    {
        LOG_ERROR("NPU 模型加载失败！");
        return -1;
    }
    
    // 初始化摔倒逻辑规则引擎
    vision::FallRuleEngine ruleEngine;

    // 视频缓存
    vision::VideoCacher videoCacher(90);

    // 实例化底层通信队列
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(3);
    concurrency::ThreadSafeQueue<vision::AlertEvent> alertQueue(10);
    
    // 线程 1：启动摄像头进行视频采集
    vision::CameraStreamer streamer(0, frameQueue, videoCacher);
    if (!streamer.start())
    {
        LOG_ERROR("摄像头启动失败！");
        return -1;
    }

    std::atomic<bool> systemRunning{true};

    // 线程 2：启动报警响应线程
    std::thread alertThread([&]()
    {
        LOG_INFO("网络通信线程已启动，正在监听报警事件...");
        while (systemRunning)
        {
            vision::AlertEvent event;

            // 阻塞等待，只有发生摔倒事件才会唤醒此线程
            alertQueue.wait_and_pop(event);

            if (!systemRunning) break;

            if (event.isFall)
            {
                LOG_WARN("处理报警事件！触发时间戳：{}", event.timestamp);

                // 触发本地蜂鸣器
                buzzer.triggerAlarm(3000);

                // Store-and-Forward 机制
                if (mqttClient.isConnected())
                {
                    // 网络在线，尝试续传历史积压报警事件
                    auto pendingAlerts = db.getPendingAlerts();
                    for (const auto& pending : pendingAlerts)
                    {
                        if (mqttClient.publishAlert("fall_detection/alerts", pending))
                        {
                            db.markAsUploaded(pending.dbId);
                        }
                    }

                    // 发布当前的最新报警
                    if (mqttClient.publishAlert("fall_detection/alerts", event))
                    {
                        LOG_INFO("报警已成功上传至云端！附带现场视频路径：{}", event.videoPath);
                    }
                    else
                    {
                        // 发送意外则本地保存
                        db.saveAlert(event);
                    }
                }
                else
                {
                    db.saveAlert(event);
                }
            }
        }
    });

    // 线程 3：主线程 AI 视觉与逻辑流水线
    LOG_INFO("主线程已启动，系统开始运行...");

    while (systemRunning)
    {
        cv::Mat currentFrame;
        frameQueue.wait_and_pop(currentFrame);
        
        if (!currentFrame.empty())
        {
            std::vector<DetectResult> aiResults;

            // NPU 特征推理
            if (inferencer.detect(currentFrame, aiResults))
            {
                vision::AlertEvent event;
                
                // 摔倒判断（高宽比 + 时序下坠速度）
                if (ruleEngine.processFrame(aiResults, event))
                {
                    LOG_WARN("摔倒事件发生！启动视频截取...");

                    // 异步保存前 3 秒视频
                    std::string videoName = "fall_record_" + std::to_string(event.timestamp) + ".mp4";
                    videoCacher.saveVideoAsync(videoName);
                    event.videoPath = videoName;

                    // 将事件放入队列交由后台线程处理
                    alertQueue.push(event);
                }
            }
        }
    }

    // 关闭系统
    streamer.stop();

    vision::AlertEvent dummyEvent;
    dummyEvent.isFall = false;
    alertQueue.push(dummyEvent);        // 唤醒并终止预警线程
    if (alertThread.joinable())
    {
        alertThread.join();
    }

    mqttClient.disconnect();
    LOG_INFO("系统资源释放完毕，安全退出！");
    return 0;
}