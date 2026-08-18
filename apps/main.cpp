#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <memory>
#include <string>
#include <csignal>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
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

// 全局运行标志：供信号处理器置 0，主循环与报警线程据此安全退出
volatile std::sig_atomic_t g_running = 1;

void handleSignal(int sig)
{
    (void)sig;
    g_running = 0;   // 信号处理器内只做标志位翻转，不进行任何加锁/内存分配
}

// 获取当前可执行文件所在目录（Linux 下通过 /proc/self/exe）
std::string getExecutableDir()
{
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
    {
        return ".";
    }
    buf[len] = '\0';
    std::string exePath(buf);
    size_t pos = exePath.find_last_of('/');
    return (pos == std::string::npos) ? "." : exePath.substr(0, pos);
}

int main(int argc, char** argv)
{
    // 初始化全局日志系统：日志写到可执行文件同级目录下的 logs/ 子目录
    std::string exeDir = getExecutableDir();
    std::string logDir = exeDir + "/logs";
    mkdir(logDir.c_str(), 0755);   // 目录已存在则忽略错误
    utils::SysLogger::getInstance().init(logDir + "/gateway.log");
    LOG_INFO("系统启动！");

    // 注册退出信号，保证 Ctrl+C / kill 时能优雅退出并刷盘日志
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    // 初始化底层硬件与容灾模块
    hardware::BuzzerController buzzer(138);
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
    
    // 线程 1：根据启动参数选择视频源
    // 用法：无参数 -> 摄像头设备 0；带文件路径参数 -> 回放本地视频（模拟回归测试）
    std::unique_ptr<vision::CameraStreamer> streamer;
    if (argc >= 2)
    {
        streamer = std::make_unique<vision::CameraStreamer>(std::string(argv[1]), frameQueue, videoCacher);
    }
    else
    {
        streamer = std::make_unique<vision::CameraStreamer>(0, frameQueue, videoCacher);
    }

    if (!streamer->start())
    {
        LOG_ERROR("视频源启动失败！");
        return -1;
    }

    // 线程 2：启动报警响应线程
    std::thread alertThread([&]()
    {
        LOG_INFO("网络通信线程已启动，正在监听报警事件...");
        while (g_running)
        {
            vision::AlertEvent event;

            // 带超时阻塞等待，超时后回到循环检查退出标志
            if (!alertQueue.wait_for_and_pop(event, std::chrono::milliseconds(500)))
            {
                continue;
            }

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

    while (g_running)
    {
        cv::Mat currentFrame;
        if (!frameQueue.wait_for_and_pop(currentFrame, std::chrono::milliseconds(500)))
        {
            continue;
        }

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
    streamer->stop();

    if (alertThread.joinable())
    {
        alertThread.join();
    }

    mqttClient.disconnect();
    LOG_INFO("系统资源释放完毕，安全退出！");
    return 0;
}