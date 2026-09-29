#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <csignal>
#include <unistd.h>
#include <limits.h>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

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
#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/network/LiveStreamer.hpp"
#include "fall-detection/vision/SafeZoneManager.hpp"
#include "fall-detection/vision/PersonTracker.hpp"
#include "fall-detection/event/EventManager.hpp"
#include "fall-detection/audio/AudioCapture.hpp"
#include "fall-detection/audio/KeywordSpotter.hpp"
#include "fall-detection/utils/SystemMonitor.hpp"

// #include <syslog.h>

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

    // 锚定系统工作目录，解决 Systemd 后台运行的路径问题
    char exePath[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if (count != -1)
    {
        std::string path(exePath, count);

        // 可执行文件在 bin/ 目录下，所以截取两次获取到的项目根目录
        std::string binDir = path.substr(0, path.find_last_of('/'));
        std::string rootDir = binDir + "/..";

        // 强制切换到当前工作目录
        if (chdir(rootDir.c_str()) != 0)
        {
            std::cerr << "警告：无法切换工作目录到 " << rootDir << std::endl;
        }
    }

    // 1. 初始化配置与全局日志
    auto& config = utils::ConfigManager::getInstance();
    config.load("configs/config.json");
    
    utils::SysLogger::getInstance().init(config.getLogFilePath());
    utils::SysLogger::getInstance().setLevel(config.getLogLevel());
    
    LOG_INFO("==================================================");
    LOG_INFO("边缘网关系统启动");
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

    network::LiveStreamer liveStreamer(640, 480, 30);
    network::MqttClient mqttClient(config.getMqttBroker(), config.getDeviceId(), config.getKeepAliveSeconds());

    // 注册 MQTT 信息回调，处理小程序发来的指令
    mqttClient.setMessageCallback([&](const std::string& topic, const std::string& payload)
    {
        try
        {
            auto json = nlohmann::json::parse(payload);
            if (json.contains("cmd"))
            {
                std::string cmd = json["cmd"];
                if (cmd == "start_live")
                {
                    std::string url = json["rtmp_url"];
                    LOG_INFO("收到小程序请求，准备推流至：{}", url);
                    liveStreamer.start(url);
                }
                else if(cmd == "stop_live")
                {
                    LOG_INFO("收到小程序请求，停止推流");
                    liveStreamer.stop();
                }
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("解析 MQTT 指令失败：{}", e.what());
        }
        
    });

    // 连接云端
    if (!mqttClient.connect())
    {
        LOG_CRITICAL("MQTT 云端连接失败！系统将暂时依赖本地容灾模块维持运行！");
    }
    else
    {
        std::string cmdTopic =
            "fall_detection/commands/" +
            config.getDeviceId();

        mqttClient.subscribe(cmdTopic);
    }

    // 3. 初始化核心视觉大脑 (修复：核心组件失败必须熔断拦截)
    vision::RKNNInferencer inferencer(config.getRknnModelPath(), config.getConfidenceThreshold(), config.getNmsThreshold());
    if (!inferencer.init()) 
    {
        LOG_ERROR("致命错误：NPU 硬件加速推理模型加载失败，系统即将强制退出！");
        return -1;
    }

    // 加载场景安全区域
    vision::SafeZoneManager safeZoneManager;
    if (!safeZoneManager.load("configs/scene_config.json"))
    {
        LOG_WARN("安全区域配置加载失败，系统暂时不启用安全躺卧区域");
    }
    else
    {
        LOG_INFO("安全躺卧区域加载完成，共 {} 个区域", safeZoneManager.getZoneCount());
    }

    vision::FallRuleConfig ruleConfig;

    // 关键点置信度
    ruleConfig.kptConfThreshold =
        config.getKptConfThreshold();

    // 姿态角度
    ruleConfig.fallAngleThreshold =
        config.getFallAngleThreshold();

    ruleConfig.recoveryAngleThreshold =
        config.getRecoveryAngleThreshold();

    // 运动特征
    ruleConfig.normalizedVelocityThreshold =
        config.getNormalizedVelocityThreshold();

    ruleConfig.normalizedCenterVelocityThreshold =
        config.getNormalizedCenterVelocityThreshold();

    // 时间状态机
    ruleConfig.suspectConfirmMs =
        config.getSuspectConfirmMs();

    ruleConfig.staticLieConfirmMs =
        config.getStaticLieConfirmMs();

    ruleConfig.fallEventWindowMs =
        config.getFallEventWindowMs();

    ruleConfig.safeZoneOverlapThreshold = config.getSafeZoneOverlapThreshold();

    vision::FallRuleEngine ruleEngine(ruleConfig, &safeZoneManager);

    vision::PersonTrackerConfig trackerConfig;

    trackerConfig.iouThreshold = 0.20f;
    trackerConfig.centerDistanceThreshold = 0.80f;
    trackerConfig.maxMissingMs = 1500;

    vision::PersonTracker personTracker(
        trackerConfig
    );

    // 离线语音初始化
    audio::KeywordSpotter keywordSpotter;
    audio::KeywordSpotterConfig kwsConfig;
    kwsConfig.encoderPath =
        "models/kws/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";

    kwsConfig.decoderPath =
        "models/kws/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";

    kwsConfig.joinerPath =
        "models/kws/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx";

    kwsConfig.tokensPath =
        "models/kws/tokens.txt";

    kwsConfig.keywordsPath =
        "models/kws/keywords_custom.txt";

    kwsConfig.sampleRate = 16000;
    kwsConfig.numThreads = 2;

    bool kwsAvailable =
        keywordSpotter.initialize(kwsConfig);


    if (!kwsAvailable)
    {
        LOG_ERROR(
            "语音关键词识别初始化失败，"
            "系统将以纯视觉模式继续运行"
        );
    }
    else
    {
        LOG_INFO(
            "离线语音关键词识别模块初始化成功"
        );
    }

    auto lastVoiceTriggerTime = std::chrono::steady_clock::time_point{};
    audio::AudioCapture audioCapture;
    event::EventManager eventManager(
        config.getDeviceId(),
        config.getDeploymentArea(),
        config.getVideoOutputDir()
    );

    // 4. 初始化流水线通信基础设施
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(config.getFrameQueueSize());
    concurrency::ThreadSafeQueue<vision::AlertEvent> alertQueue(config.getAlertQueueSize());
    vision::VideoCacher videoCacher(config.getVideoCacheFrames());

    if (kwsAvailable)
    {
        audio::AudioCaptureConfig audioConfig;

        /*
        * 前面通过 arecord -l 已经确认：
        *
        * card 2
        * device 0
        */
        audioConfig.device =
            "plughw:CARD=Device,DEV=0";

        audioConfig.sampleRate =
            16000;

        audioConfig.channels =
            1;

        /*
        * 100ms 音频：
        *
        * 16000 × 0.1 = 1600 samples
        */
        audioConfig.framesPerChunk =
            1600;


        const bool audioStarted =
            audioCapture.start(
                audioConfig,

                [&](const std::vector<float>& samples)
                {
                    auto keyword =
                        keywordSpotter.processSamples(
                            samples
                        );


                    if (keyword)
                    {
                        const auto now =
                            std::chrono::steady_clock::now();


                        const auto elapsedMs =
                            std::chrono::duration_cast<
                                std::chrono::milliseconds
                            >(
                                now - lastVoiceTriggerTime
                            ).count();


                        constexpr long long VOICE_COOLDOWN_MS =
                            5000;


                        /*
                        * 第一次触发，或者距离上次已经超过5秒。
                        */
                        if (lastVoiceTriggerTime.time_since_epoch().count() == 0 ||
                            elapsedMs >= VOICE_COOLDOWN_MS)
                        {
                            lastVoiceTriggerTime =
                                now;


                            LOG_WARN(
                                "【语音求救】检测到关键词：{}",
                                *keyword
                            );


                            fall_detection::event::AlertEvent voiceEvent;

                            voiceEvent.eventType =
                                fall_detection::event::EventType::HELP_REQUEST;

                            voiceEvent.source = {
                                fall_detection::event::EventSource::VOICE
                            };

                            voiceEvent.status =
                                fall_detection::event::EventStatus::NEW;

                            voiceEvent.keyword =
                                *keyword;

                            voiceEvent.timestamp =
                                std::chrono::duration_cast<
                                    std::chrono::milliseconds
                                >(
                                    std::chrono::system_clock::now()
                                        .time_since_epoch()
                                ).count();


                            auto managedEvent =
                                eventManager.prepareEvent(
                                    std::move(voiceEvent)
                                );


                            alertQueue.push(
                                std::move(managedEvent)
                            );
                        }
                    }
                }
            );


        if (!audioStarted)
        {
            LOG_ERROR(
                "USB 麦克风启动失败，"
                "系统将以纯视觉模式继续运行"
            );
        }
    }

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

    // 将推流器绑定到采集线程，实现解耦
    streamer->setLiveStreamer(&liveStreamer);

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
                if (event.captureVideo && !event.videoPath.empty())
                {
                    int fps = config.getVideoSaveFps();

                    videoCacher.saveVideoAsync(
                        event.videoPath,
                        fps,
                        fps * 5
                    );
                }
                else if (
                    event.eventType ==
                    fall_detection::event::EventType::HELP_REQUEST)
                {
                    LOG_WARN(
                        "收到语音求救报警："
                        "event_id={}, keyword={}",
                        event.eventId,
                        event.keyword
                    );
                }

                if (!event.isFusionUpdate)
                {
                    buzzer.triggerAlarm(config.getAlarmDurationMs());
                }

                // MQTT 上报
                if (mqttClient.isConnected())
                {
                    // 补传历史报警
                    auto pendingAlerts = localDb.getPendingAlerts();
                    for (auto& pendingEvent : pendingAlerts)
                    {
                        LOG_INFO("准备补传历史报警: event_id={}, type={}", pendingEvent.eventId, event::toString(pendingEvent.eventType));

                        if (mqttClient.publishAlert(config.getAlertTopic(), pendingEvent))
                        {
                            localDb.markAsUploaded(pendingEvent.dbId);
                        }
                    }

                    LOG_INFO("MQTT在线, 准备上报: event_id={}, type={}", event.eventId, fall_detection::event::toString(event.eventType));
                    
                    if (!mqttClient.publishAlert(config.getAlertTopic(), event))
                    {
                        LOG_ERROR("MQTT报警发送失败: event_id={}", event.eventId);
                        localDb.saveAlert(event);
                    }
                }
                else
                {
                    LOG_WARN("MQTT当前离线,暂时无法上报警报: event_id={}", event.eventId);
                    localDb.saveAlert(event);
                }
            }
        }
    });

    // 系统硬件心跳守护进程
    // 负责 10 秒向云端发送存活证明
    utils::SystemMonitor systemMonitor;
    std::thread heartbeatThread([&]()
    {
        LOG_INFO("[Heartbeat] 心跳线程启动");
        while (g_running)
        {
            if (mqttClient.isConnected())
            {
                LOG_INFO("[Heartbeat] 开始采集系统资源");
                int cpuUsage = systemMonitor.getCpuUsage();
                int memoryUsage = systemMonitor.getMemoryUsage();
                int storageUsage = systemMonitor.getStorageUsage("/");

                std::string statusTopic = "fall_detection/status/" + config.getDeviceId();

                LOG_INFO(
                    "设备状态心跳：CPU={}%，内存={}%，存储={}%",
                    cpuUsage,
                    memoryUsage,
                    storageUsage
                );

                bool ok = mqttClient.publishStatus(
                    statusTopic,
                    cpuUsage,
                    memoryUsage,
                    storageUsage,
                    config.getDeploymentArea()
                );

                LOG_INFO("[Heartbeat] MQTT 心跳发送调试结束, result={}", ok);

            }

            for (int i = 0; i < 100 && g_running; ++i)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        LOG_INFO("[Heartbeat] 心跳线程已退出");
    });

    // 7. AI 主干视觉流水线 (生产者)
    LOG_INFO("AI 视觉主干流水线已就绪，进入实时监测模式...");
    // ==================== 性能统计 ====================
    constexpr int PERF_WARMUP_FRAMES = 20;
    constexpr double PERF_REPORT_INTERVAL_SEC = 5.0;

    int warmupFrames = 0;

    int perfFrames = 0;
    double perfDetectTotalMs = 0.0;
    double perfDetectMinMs = 1e9;
    double perfDetectMaxMs = 0.0;

    auto perfWindowStart = std::chrono::steady_clock::now();
    // ==================================================
    while (g_running)
    {
        cv::Mat frame;

        if (!frameQueue.wait_for_and_pop(
                frame, std::chrono::milliseconds(500)))
        {
            continue;
        }

        // ==================== detect 耗时统计 ====================
        auto detectStart = std::chrono::steady_clock::now();

        std::vector<vision::DetectResult> aiResults;
        bool detectOk = inferencer.detect(frame, aiResults);

        auto trackedPersons = personTracker.update(aiResults);
        for (const auto& person : trackedPersons)
        {
            const auto& det =
                person.detection;

            LOG_TRACE(
                "[PersonTracker] ID={}, "
                "bbox=({}, {}, {}, {}), conf={:.2f}",
                person.trackId,
                det.x,
                det.y,
                det.width,
                det.height,
                det.confidence
            );
        }

        auto detectEnd = std::chrono::steady_clock::now();

        double detectMs =
            std::chrono::duration<double, std::milli>(
                detectEnd - detectStart).count();
        // =========================================================

        if (detectOk)
        {
            // 前 20 帧作为预热帧，不计入性能统计
            if (warmupFrames < PERF_WARMUP_FRAMES)
            {
                ++warmupFrames;

                if (warmupFrames == PERF_WARMUP_FRAMES)
                {
                    perfWindowStart = std::chrono::steady_clock::now();

                    LOG_INFO("[PERF] 预热完成，开始统计 AI 实际性能...");
                }
            }
            else
            {
                ++perfFrames;

                perfDetectTotalMs += detectMs;

                if (detectMs < perfDetectMinMs)
                {
                    perfDetectMinMs = detectMs;
                }

                if (detectMs > perfDetectMaxMs)
                {
                    perfDetectMaxMs = detectMs;
                }
            }

            auto fallEvents =
                ruleEngine.processFrame(
                    trackedPersons,
                    frame.cols,
                    frame.rows
                );

            for (auto& fallEvent : fallEvents)
            {
                /*
                * FallRuleEngine 只负责判断：
                * “发生了跌倒”
                *
                * EventManager 负责补充：
                * event_id
                * device_id
                */
                auto managedEvent =
                    eventManager.prepareEvent(
                        std::move(fallEvent)
                    );


                LOG_WARN(
                    "报警事件进入队列："
                    "event_id={}, Track={}",
                    managedEvent.eventId,
                    managedEvent.personTrackId
                );


                alertQueue.push(
                    std::move(managedEvent)
                );
            }
        }

        // ==================== 每 5 秒输出一次性能 ====================
        if (warmupFrames >= PERF_WARMUP_FRAMES && perfFrames > 0)
        {
            auto now = std::chrono::steady_clock::now();

            double elapsedSec =
                std::chrono::duration<double>(
                    now - perfWindowStart).count();

            if (elapsedSec >= PERF_REPORT_INTERVAL_SEC)
            {
                double fps = perfFrames / elapsedSec;
                double avgDetectMs =
                    perfDetectTotalMs / perfFrames;

                LOG_INFO(
                    "[PERF] AI pipeline: FPS={:.2f}, detect avg={:.2f} ms, min={:.2f} ms, max={:.2f} ms, frames={}",
                    fps,
                    avgDetectMs,
                    perfDetectMinMs,
                    perfDetectMaxMs,
                    perfFrames
                );

                // 重置当前统计窗口
                perfWindowStart = now;
                perfFrames = 0;
                perfDetectTotalMs = 0.0;
                perfDetectMinMs = 1e9;
                perfDetectMaxMs = 0.0;
            }
        }
        // =========================================================
    }

    // 8. 捕获信号并执行优雅退出
    LOG_INFO("接收到退出信号，正在安全释放所有系统组件...");

    LOG_INFO("[Shutdown] 正在停止 CameraStreamer...");
    streamer->stop();
    LOG_INFO("[Shutdown] CameraStreamer 已停止");

    LOG_INFO("[Shutdown] 正在停止 AudioCapture...");
    audioCapture.stop();
    LOG_INFO("[Shutdown] AudioCapture 已停止");

    LOG_INFO("[Shutdown] 正在停止 LiveStreamer...");
    liveStreamer.stop();
    LOG_INFO("[Shutdown] LiveStreamer 已停止");

    LOG_INFO("[Shutdown] 正在等待 heartbeatThread...");
    if (heartbeatThread.joinable())
    {
        heartbeatThread.join();
    }
    LOG_INFO("[Shutdown] heartbeatThread 已退出");

    LOG_INFO("[Shutdown] 正在等待 alertThread...");
    if (alertThread.joinable())
    {
        alertThread.join();
    }
    LOG_INFO("[Shutdown] alertThread 已退出");

    LOG_INFO("[Shutdown] 正在断开 MQTT...");
    mqttClient.disconnect();
    LOG_INFO("[Shutdown] MQTT 已断开");

    LOG_INFO("系统资源释放完毕，安全退出！");
    return 0;
}