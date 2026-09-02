#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <csignal>

#include "fall-detection/utils/ConfigManager.hpp"
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

// 信号驱动的优雅退出标志
volatile std::sig_atomic_t g_running = 1;
void signalHandler(int signum) 
{
    g_running = 0;
}

int main(int argc, char* argv[])
{
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 1. 初始化配置与全局日志
    auto& config = utils::ConfigManager::getInstance();
    config.load("configs/config.json");
    
    utils::SysLogger::getInstance().init(config.getLogFilePath());
    utils::SysLogger::getInstance().setLevel(config.getLogLevel());
    
    LOG_INFO("==================================================");
    LOG_INFO("边缘网关系统启动 - 企业级高可用容灾版");
    LOG_INFO("==================================================");

    // 2. 初始化降级容灾模块 (修复：增加严格的状态校验与降级告警)
    hardware::BuzzerController buzzer(config.getBuzzerGpioPin());
    if (!buzzer.init()) 
    {
        LOG_CRITICAL("【降级警告】蜂鸣器物理容灾模块初始化失败！断网时将无法发出声光报警！");
    }

    utils::LocalDatabase localDb(config.getSqliteDbPath());
    if (!localDb.init()) 
    {
        LOG_CRITICAL("【降级警告】本地 SQLite 容灾数据库初始化失败！断网时报警数据将面临丢失风险！");
    }

    network::MqttClient mqttClient(config.getMqttBroker(), config.getMqttClientId(), config.getKeepAliveSeconds());
    if (!mqttClient.connect()) 
    {
        LOG_CRITICAL("【降级警告】MQTT 云端连接失败！系统将暂时依赖本地容灾模块维持运行！");
    }

    // 3. 初始化核心视觉大脑 (修复：核心组件失败必须熔断拦截)
    vision::RKNNInferencer inferencer(config.getRknnModelPath(), config.getConfidenceThreshold(), config.getNmsThreshold());
    if (!inferencer.init()) 
    {
        LOG_ERROR("致命错误：NPU 硬件加速推理模型加载失败，系统即将强制退出！");
        return -1;
    }

    vision::FallRuleConfig ruleConfig;
    ruleConfig.fallAngleThreshold = config.getFallAngleThreshold();
    ruleConfig.fallVelocityThreshold = config.getFallVelocityThreshold();
    ruleConfig.confirmFramesThreshold = config.getConfirmFramesThreshold();
    ruleConfig.staticLieThreshold = config.getStaticLieThreshold();
    ruleConfig.fallEventWindow = config.getFallEventWindow();
    vision::FallRuleEngine ruleEngine(ruleConfig);

    // 4. 初始化流水线通信基础设施
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(config.getFrameQueueSize());
    concurrency::ThreadSafeQueue<vision::AlertEvent> alertQueue(config.getAlertQueueSize());
    vision::VideoCacher videoCacher(config.getVideoCacheFrames());

    // 5. 启动“眼睛”
    std::unique_ptr<vision::CameraStreamer> streamer;
    if (argc > 1) 
    {
        streamer = std::make_unique<vision::CameraStreamer>(argv[1], frameQueue, videoCacher);
    } 
    else 
    {
        streamer = std::make_unique<vision::CameraStreamer>(config.getCameraDeviceId(), frameQueue, videoCacher);
    }

    if (!streamer->start()) 
    {
        LOG_ERROR("致命错误：视频流采集模块启动失败！");
        return -1;
    }

    // 6. 启动预警响应后台线程 (消费者)
    std::thread alertThread([&]() 
    {
        while (g_running) 
        {
            vision::AlertEvent event;
            // 采用带超时的出队，确保关机时能及时打破死锁
            if (alertQueue.wait_for_and_pop(event, std::chrono::milliseconds(500))) 
            {
                if (event.isFall) 
                {
                    LOG_INFO("报警线程响应：合成现场取证视频并联动声光告警...");
                    
                    event.videoPath = config.getVideoOutputDir() + "/fall_" + std::to_string(event.timestamp) + ".mp4";
                    videoCacher.saveVideoAsync(event.videoPath, config.getVideoSaveFps());
                    buzzer.triggerAlarm(config.getAlarmDurationMs());

                    // Store-and-Forward 容灾上报闭环
                    if (mqttClient.isConnected()) 
                    {
                        auto pendingAlerts = localDb.getPendingAlerts();
                        for (const auto& pa : pendingAlerts) 
                        {
                            if (mqttClient.publishAlert(config.getAlertTopic(), pa)) 
                            {
                                localDb.markAsUploaded(pa.dbId);
                            }
                        }
                        if (!mqttClient.publishAlert(config.getAlertTopic(), event)) 
                        {
                            localDb.saveAlert(event);
                        }
                    } 
                    else 
                    {
                        localDb.saveAlert(event);
                    }
                }
            }
        }
    });

    // 7. AI 主干视觉流水线 (生产者)
    LOG_INFO("AI 视觉主干流水线已就绪，进入实时监测模式...");
    while (g_running) 
    {
        cv::Mat frame;
        if (frameQueue.wait_for_and_pop(frame, std::chrono::milliseconds(500))) 
        {
            std::vector<vision::DetectResult> aiResults;
            if (inferencer.detect(frame, aiResults)) 
            {
                vision::AlertEvent event;
                if (ruleEngine.processFrame(aiResults, event)) 
                {
                    alertQueue.push(event);
                }
            }
        }
    }

    // 8. 捕获信号并执行优雅退出
    LOG_INFO("接收到退出信号，正在安全释放所有系统组件...");
    streamer->stop();
    if (alertThread.joinable()) 
    {
        alertThread.join();
    }
    mqttClient.disconnect();
    
    LOG_INFO("系统资源释放完毕，安全退出！");
    return 0;
}