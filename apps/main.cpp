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
#include <stdexcept>
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

// 信号驱动的退出标志
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
    
    LOG_INFO("跌倒监测系统启动");

    // 2. 初始化降级容灾模块
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

    const std::string sceneConfigPath = "configs/scene_config.json";

    vision::SafeZoneManager safeZoneManager;
    if (!safeZoneManager.load(sceneConfigPath))
    {
        LOG_WARN("安全区域配置加载失败，系统暂时不启用安全躺卧区域");
    }
    else
    {
        LOG_INFO("安全躺卧区域加载完成，共 {} 个区域", safeZoneManager.getZoneCount());
    }

    network::LiveStreamer liveStreamer(640, 480, 30);
    network::MqttClient mqttClient(config.getMqttBroker(), config.getDeviceId(), config.getKeepAliveSeconds());

    // 注册 MQTT 信息回调，处理小程序发来的指令
    mqttClient.setMessageCallback([&](const std::string& topic, const std::string& payload)
    {
        try
        {
            const auto json = nlohmann::json::parse(payload);

            if (!json.contains("cmd") || !json["cmd"].is_string())
            {
                LOG_WARN("收到无效 MQTT 指令：缺少 cmd");
                return;
            }

            const std::string cmd = json["cmd"].get<std::string>();

            if (cmd == "start_live")
            {
                if (!json.contains("rtmp_url") || !json["rtmp_url"].is_string())
                {
                    LOG_WARN("start_live 指令缺少 rtmp_url");
                    return;
                }

                const std::string url = json["rtmp_url"].get<std::string>();
                LOG_INFO("收到实时视频请求，准备推流至：{}", url);
                liveStreamer.start(url);
            }
            else if (cmd == "stop_live")
            {
                LOG_INFO("收到停止实时视频指令");
                liveStreamer.stop();
            }
            else if (cmd == "update_safe_zones")
            {
                const std::string requestId = json.value("request_id", "");

                if (!json.contains("safe_lie_zones") || !json["safe_lie_zones"].is_array())
                {
                    LOG_ERROR("安全区域配置指令格式错误：safe_lie_zones 不存在或不是数组，request_id={}", requestId);
                    return;
                }

                std::vector<vision::SafeLieZone> newZones;

                for (const auto& zoneJson : json["safe_lie_zones"])
                {
                    if (!zoneJson.is_object())
                    {
                        throw std::runtime_error("安全区域必须为 JSON 对象");
                    }

                    vision::SafeLieZone zone;
                    zone.name = zoneJson.value("name", "");

                    if (!zoneJson.contains("points") || !zoneJson["points"].is_array())
                    {
                        throw std::runtime_error("安全区域 points 不存在或不是数组");
                    }

                    for (const auto& pointJson : zoneJson["points"])
                    {
                        if (!pointJson.is_array() || pointJson.size() != 2)
                        {
                            throw std::runtime_error("安全区域坐标必须为 [x, y]");
                        }

                        vision::NormalizedPoint point;
                        point.x = pointJson[0].get<float>();
                        point.y = pointJson[1].get<float>();

                        zone.points.push_back(point);
                    }

                    newZones.push_back(std::move(zone));
                }

                if (safeZoneManager.replaceZonesAndSave(newZones, sceneConfigPath))
                {
                    LOG_INFO("安全区域配置更新成功：request_id={}, zone_count={}", requestId, safeZoneManager.getZoneCount());
                }
                else
                {
                    LOG_ERROR("安全区域配置更新失败：request_id={}", requestId);
                }
            }
            else
            {
                LOG_WARN("收到未知 MQTT 指令：{}", cmd);
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

    // 3. 初始化核心视觉大脑 
    vision::RKNNInferencer inferencer(config.getRknnModelPath(), config.getConfidenceThreshold(), config.getNmsThreshold());
    if (!inferencer.init()) 
    {
        LOG_ERROR("致命错误：NPU 硬件加速推理模型加载失败，系统即将强制退出！");
        return -1;
    }

    vision::FallRuleConfig ruleConfig;

    // 关键点置信度
    ruleConfig.kptConfThreshold = config.getKptConfThreshold();

    // 姿态角度
    ruleConfig.fallAngleThreshold = config.getFallAngleThreshold();

    ruleConfig.recoveryAngleThreshold = config.getRecoveryAngleThreshold();

    // 运动特征
    ruleConfig.normalizedVelocityThreshold = config.getNormalizedVelocityThreshold();

    ruleConfig.normalizedCenterVelocityThreshold = config.getNormalizedCenterVelocityThreshold();

    // 时间状态机
    ruleConfig.suspectConfirmMs = config.getSuspectConfirmMs();

    ruleConfig.staticLieConfirmMs = config.getStaticLieConfirmMs();

    ruleConfig.fallEventWindowMs = config.getFallEventWindowMs();

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
    kwsConfig.encoderPath = "models/kws/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
    kwsConfig.decoderPath = "models/kws/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
    kwsConfig.joinerPath = "models/kws/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx";
    kwsConfig.tokensPath = "models/kws/tokens.txt";
    kwsConfig.keywordsPath = "models/kws/keywords_custom.txt";
    kwsConfig.sampleRate = 16000;
    kwsConfig.numThreads = 2;
    bool kwsAvailable = keywordSpotter.initialize(kwsConfig);
    if (!kwsAvailable)
    {
        LOG_ERROR("语音关键词识别初始化失败，系统将以纯视觉模式继续运行");
    }
    else
    {
        LOG_INFO("离线语音关键词识别模块初始化成功");
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

        audioConfig.device = "plughw:CARD=Device,DEV=0";

        audioConfig.sampleRate = 16000;

        audioConfig.channels = 1;

        /*
        * 100ms 音频：
        *
        * 16000 × 0.1 = 1600 samples
        */
        audioConfig.framesPerChunk = 1600;

        const bool audioStarted = audioCapture.start(audioConfig,[&](const std::vector<float>& samples)
                {
                    auto keyword = keywordSpotter.processSamples(samples);

                    if (keyword)
                    {
                        const auto now = std::chrono::steady_clock::now();

                        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastVoiceTriggerTime).count();

                        constexpr long long VOICE_COOLDOWN_MS = 5000;

                        /*
                        * 第一次触发，或者距离上次已经超过5秒。
                        */
                        if (lastVoiceTriggerTime.time_since_epoch().count() == 0 || elapsedMs >= VOICE_COOLDOWN_MS)
                        {
                            lastVoiceTriggerTime = now;

                            LOG_WARN("【语音求救】检测到关键词：{}",*keyword);

                            fall_detection::event::AlertEvent voiceEvent;

                            voiceEvent.eventType = fall_detection::event::EventType::HELP_REQUEST;

                            voiceEvent.source = {fall_detection::event::EventSource::VOICE};

                            voiceEvent.status = fall_detection::event::EventStatus::NEW;

                            voiceEvent.keyword = *keyword;

                            voiceEvent.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

                            auto managedEvent = eventManager.prepareEvent(std::move(voiceEvent));

                            alertQueue.push(std::move(managedEvent));
                        }
                    }
                }
            );


        if (!audioStarted)
        {
            LOG_ERROR("USB 麦克风启动失败，系统将以纯视觉模式继续运行");
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

    // 6. 统一报警响应线程
    std::thread alertThread([&]()
    {
        LOG_INFO("[Alert] 报警处理线程已启动");

        while (g_running)
        {
            event::AlertEvent alertEvent;

            // 带超时等待，保证 Ctrl+C 后能够及时退出
            if (!alertQueue.wait_for_and_pop(
                    alertEvent,
                    std::chrono::milliseconds(500)))
            {
                continue;
            }

            LOG_WARN(
                "[Alert] 收到报警事件：event_id={}, type={}, fusion_update={}",
                alertEvent.eventId,
                event::toString(alertEvent.eventType),
                alertEvent.isFusionUpdate
            );

            // 1. 本地蜂鸣报警
            // 首次报警才触发蜂鸣，融合更新不重复鸣叫。
            if (!alertEvent.isFusionUpdate)
            {
                buzzer.triggerAlarm(config.getAlarmDurationMs());
            }

            // 2. 现场视频
            // 只有包含视觉跌倒时才录像：
            if (alertEvent.captureVideo)
            {
                if (alertEvent.videoPath.empty())
                {
                    alertEvent.videoPath = config.getVideoOutputDir() + "/fall_" + alertEvent.eventId + ".mp4";
                }

                LOG_INFO(
                    "[Alert] 启动事件录像：event_id={}, path={}",
                    alertEvent.eventId,
                    alertEvent.videoPath
                );

                // 这里保持你目前已经实现成功的
                // “预录约3秒 + 后录约5秒”调用方式。
                videoCacher.saveVideoAsync(alertEvent.videoPath,config.getVideoSaveFps());
            }

            // 3. Local First
            // 不管网络是否正常，报警首先写入 SQLite。
            // saveAlert 内部使用 event_id UPSERT：
            // 第一次事件 -> INSERT PENDING
            // 融合事件   -> UPDATE 同一 event_id，重新 PENDING
            if (!localDb.saveAlert(alertEvent))
            {
                LOG_ERROR(
                    "[Alert] 本地持久化失败：event_id={}",
                    alertEvent.eventId
                );
            }

            // 4. 当前网络在线则立即尝试上传
            if (mqttClient.isConnected())
            {
                LOG_INFO(
                    "[Alert] MQTT 在线，立即上报：event_id={}",
                    alertEvent.eventId
                );

                if (mqttClient.publishAlert(config.getAlertTopic(), alertEvent))
                {
                    // publishAlert 返回 true 表示
                    // QoS 1 已收到 Broker 确认。
                    localDb.markAsUploaded(
                        alertEvent.eventId
                    );

                    LOG_INFO(
                        "[Alert] 报警上报成功：event_id={}",
                        alertEvent.eventId
                    );
                }
                else
                {
                    LOG_WARN(
                        "[Alert] 报警上报失败，保留 PENDING 等待补传：event_id={}",
                        alertEvent.eventId
                    );
                }
            }
            else
            {
                // 同样不需要再调用 saveAlert，
                // 因为前面已经保存过了。
                LOG_WARN(
                    "[Alert] MQTT 当前离线，报警已本地保存：event_id={}",
                    alertEvent.eventId
                );
            }
        }

        LOG_INFO("[Alert] 报警处理线程已退出");
    });

    std::thread retryThread([&]()
    {
        LOG_INFO("[Retry] 报警补传线程已启动");

        while (g_running)
        {
            if (mqttClient.isConnected())
            {
                auto pendingAlerts = localDb.getPendingAlerts();

                if (!pendingAlerts.empty())
                {
                    LOG_INFO(
                        "[Retry] 检测到 {} 条待补传报警",
                        pendingAlerts.size()
                    );
                }

                for (const auto& alert : pendingAlerts)
                {
                    if (!g_running)
                    {
                        break;
                    }

                    LOG_INFO(
                        "[Retry] 正在补传报警：event_id={}",
                        alert.eventId
                    );

                    if (mqttClient.publishAlert(config.getAlertTopic(),alert))
                    {
                        localDb.markAsUploaded(alert.eventId);

                        LOG_INFO(
                            "[Retry] 报警补传成功：event_id={}",
                            alert.eventId
                        );
                    }
                    else
                    {
                        LOG_WARN("[Retry] 报警补传失败，等待下次重试：event_id={}",alert.eventId);

                        // 网络可能刚刚再次断开，
                        // 本轮不继续连续发送其他记录
                        break;
                    }
                }
            }

            // 每 3 秒检查一次，但 100ms 就能响应退出
            for (int i = 0;
                i < 30 && g_running;
                ++i)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100)
                );
            }
        }

        LOG_INFO("[Retry] 报警补传线程已退出");
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
    //  性能统计 
    constexpr int PERF_WARMUP_FRAMES = 20;
    constexpr double PERF_REPORT_INTERVAL_SEC = 5.0;

    int warmupFrames = 0;

    int perfFrames = 0;
    double perfDetectTotalMs = 0.0;
    double perfDetectMinMs = 1e9;
    double perfDetectMaxMs = 0.0;

    auto perfWindowStart = std::chrono::steady_clock::now();
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

    LOG_INFO("[Shutdown] 正在等待 retryThread...");
    if (retryThread.joinable())
    {
        retryThread.join();
    }
    LOG_INFO("[Shutdown] retryThread 已退出");
    LOG_INFO("[Shutdown] 正在断开 MQTT...");
    mqttClient.disconnect();
    LOG_INFO("[Shutdown] MQTT 已断开");

    LOG_INFO("系统资源释放完毕，安全退出！");
    return 0;
}