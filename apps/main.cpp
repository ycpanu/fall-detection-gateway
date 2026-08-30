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
#include "fall-detection/utils/ConfigManager.hpp"
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

// 相对路径统一基于可执行文件目录解析，绝对路径原样返回
std::string resolvePath(const std::string& baseDir, const std::string& path)
{
    if (path.empty() || path[0] == '/')
    {
        return path;
    }
    return baseDir + "/" + path;
}

// 确保某个文件的父目录存在（尽力而为，已存在则忽略）
void ensureParentDir(const std::string& filePath)
{
    size_t pos = filePath.find_last_of('/');

    // 没找到 '/' 会返回std::string::npos
    if (pos == std::string::npos)
    {
        return;
    }
    std::string dir = filePath.substr(0, pos);
    if (!dir.empty())
    {
        mkdir(dir.c_str(), 0755);
    }
}

int main(int argc, char** argv)
{
    std::string exeDir = getExecutableDir();

    // 1. 加载全局配置（此时日志系统尚未初始化，ConfigManager 内部错误只打印到 stderr）
    utils::ConfigManager& config = utils::ConfigManager::getInstance();
    bool configOk = config.load("configs/config.json");

    // 2. 初始化日志系统：路径与级别均来自配置
    std::string logPath = resolvePath(exeDir, config.getLogFilePath());
    ensureParentDir(logPath);
    utils::SysLogger::getInstance().init(logPath);
    utils::SysLogger::getInstance().setLevel(config.getLogLevel());

    if (configOk)
    {
        LOG_INFO("配置文件加载成功！");
    }
    else
    {
        LOG_ERROR("配置文件加载失败，将使用内置默认参数继续运行！");
    }
    LOG_INFO("系统启动！");

    // 注册退出信号，保证 Ctrl+C / kill 时能退出并刷盘日志
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    // 初始化底层硬件与容灾模块
    hardware::BuzzerController buzzer(config.getBuzzerGpioPin());
    buzzer.init();

    utils::LocalDatabase db(config.getSqliteDbPath());
    db.init();

    // 初始化网络通信层
    network::MqttClient mqttClient(config.getMqttBroker(), config.getMqttClientId(), config.getKeepAliveSeconds());
    mqttClient.connect();

    // 初始化 NPU 硬件推理
    vision::RKNNInferencer inferencer(config.getRknnModelPath(),
                                      config.getConfidenceThreshold(),
                                      config.getNmsThreshold());
    if (!inferencer.init())
    {
        LOG_ERROR("NPU 模型加载失败！");
        return -1;
    }

    // 初始化摔倒逻辑规则引擎
    vision::FallRuleConfig ruleConfig;
    ruleConfig.kptConfThreshold = config.getKptConfThreshold();
    ruleConfig.fallAngleThreshold = config.getFallAngleThreshold();
    ruleConfig.fallVelocityThreshold = config.getFallVelocityThreshold();
    ruleConfig.confirmFramesThreshold = config.getConfirmFramesThreshold();
    ruleConfig.staticLieThreshold = config.getStaticLieThreshold();
    ruleConfig.fallEventWindow = config.getFallEventWindow();
    vision::FallRuleEngine ruleEngine(ruleConfig);

    // 视频缓存
    vision::VideoCacher videoCacher(config.getVideoCacheFrames());

    // 实例化底层通信队列
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(config.getFrameQueueSize());
    concurrency::ThreadSafeQueue<vision::AlertEvent> alertQueue(config.getAlertQueueSize());

    // 视频落盘目录
    std::string videoDir = resolvePath(exeDir, config.getVideoOutputDir());
    mkdir(videoDir.c_str(), 0755);

    // 线程 1：根据启动参数选择视频源
    // 用法：无参数 -> 摄像头设备；带文件路径参数 -> 回放本地视频（模拟回归测试）
    std::unique_ptr<vision::CameraStreamer> streamer;
    if (argc >= 2)
    {
        streamer = std::make_unique<vision::CameraStreamer>(std::string(argv[1]), frameQueue, videoCacher);
    }
    else
    {
        streamer = std::make_unique<vision::CameraStreamer>(config.getCameraDeviceId(), frameQueue, videoCacher);
    }

    if (!streamer->start())
    {
        LOG_ERROR("视频源启动失败！");
        return -1;
    }

    // 线程 2：启动报警响应线程
    std::string alertTopic = config.getAlertTopic();
    int alarmDurationMs = config.getAlarmDurationMs();
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
                buzzer.triggerAlarm(alarmDurationMs);

                // Store-and-Forward 机制
                if (mqttClient.isConnected())
                {
                    // 网络在线，尝试续传历史积压报警事件
                    auto pendingAlerts = db.getPendingAlerts();
                    for (const auto& pending : pendingAlerts)
                    {
                        if (mqttClient.publishAlert(alertTopic, pending))
                        {
                            db.markAsUploaded(pending.dbId);
                        }
                    }

                    // 发布当前的最新报警
                    if (mqttClient.publishAlert(alertTopic, event))
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
            std::vector<vision::DetectResult> aiResults;

            // NPU 特征推理
            if (inferencer.detect(currentFrame, aiResults))
            {
                vision::AlertEvent event;

                // 摔倒判断（高宽比 + 时序下坠速度）
                if (ruleEngine.processFrame(aiResults, event))
                {
                    LOG_WARN("摔倒事件发生！启动视频截取...");

                    // 异步保存前 N 秒视频
                    std::string videoName = videoDir + "/fall_record_" + std::to_string(event.timestamp) + ".mp4";
                    videoCacher.saveVideoAsync(videoName, config.getVideoSaveFps());
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
