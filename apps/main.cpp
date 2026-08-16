#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/concurrency/ThreadSafeQueue.hpp"
#include "fall-detection/vision/CameraStreamer.hpp"
#include "fall-detection/vision/FallRuleEngine.hpp"
#include "fall-detection/vision/RKNNInferencer.hpp"
#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/network/MqttClient.hpp"

using namespace fall_detection;

int main(int argc, char** argv)
{
    // 1. 初始化全局日志系统
    utils::SysLogger::getInstance().init("logs/gateway.log");
    LOG_INFO("系统启动！");
    
    // 初始化网络通信层
    network::MqttClient mqttClient("tcp://broker.emqx.io:1883", "Orangepi_Gateway_001");
    mqttClient.connect();

    // 2. 初始化 NPU 硬件推理
    RKNNInferencer inferencer("./best.rknn");
    if (!inferencer.init())
    {
        LOG_ERROR("NPU 模型加载失败！");
        return -1;
    }
    
    // 3. 初始化摔倒逻辑规则引擎
    vision::FallRuleEngine ruleEngine;

    // 4. 实例化底层通信
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(3);
    concurrency::ThreadSafeQueue<vision::AlertEvent> alertQueue(10);
    
    // 5. 启动摄像头
    vision::VideoCacher videoCacher(90);
    vision::CameraStreamer streamer(0, frameQueue, videoCacher);
    if (!streamer.start())
    {
        LOG_ERROR("摄像头启动失败！");
        return -1;
    }

    std::atomic<bool> systemRunning{true};
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
                LOG_INFO("通信线程收到报警！准备数据打包 JSON 上传云端...");

                // 负责将 JSON 数据打包上传，同时驱动本地硬件报警
                // 将 JSON 数据发送到特定主题
                mqttClient.publishAlert("fall_gateway/alerts", event);

                // 后续加入本地蜂鸣器
            }
        }
    });

    LOG_INFO("主线程已启动");

    int testFrameCount = 0;
    const int MAX_TEST_FRAMES = 500;
    
    while (testFrameCount < MAX_TEST_FRAMES)
    {
        cv::Mat currentFrame;
        
        // 阻塞等待采集线程抓取的新画面
        frameQueue.wait_and_pop(currentFrame);
        
        if (!currentFrame.empty())
        {
            testFrameCount++;
            
            // 进入 NPU 进行极速特征推理
            std::vector<DetectResult> aiResults;
            if (inferencer.detect(currentFrame, aiResults))
            {
                // 将 AI 输出的静态特征送入规则引擎判断
                vision::AlertEvent event;
                bool isFallConfirmed = ruleEngine.processFrame(aiResults, event);

                // 如果连续命中 lie 状态，且符合 W > H 和下坠速度阈值
                if (isFallConfirmed)
                {
                    LOG_WARN("摔倒报警！触发中心点坐标：({},{})", event.triggerBoxX, event.triggerBoxY);

                }
            }
        }
    }

    // 6. 关闭系统
    systemRunning = false;

    streamer.stop();

    vision::AlertEvent dummyEvent;
    dummyEvent.isFall = false;
    alertQueue.push(dummyEvent);
    if (alertThread.joinable())
    {
        alertThread.join();
    }
    
    mqttClient.disconnect();
    
    LOG_INFO("系统关闭！");

    return 0;
}