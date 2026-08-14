#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/concurrency/ThreadSafeQueue.hpp"
#include "fall-detection/vision/CameraStreamer.hpp"
#include "fall-detection/vision/RKNNInferencer.hpp"

using namespace fall_detection;

int main(int argc, char* argv[])
{
    // 1. 初始化全局日志系统
    utils::SysLogger::getInstance().init("logs/gateway.log");

    LOG_INFO("=======================================");
    LOG_INFO("系统启动");
    LOG_INFO("=======================================");

    // 2. 加载 NPU 模型
    RKNNInferencer inferencer("./best.rknn");
    if (!inferencer.init())
    {
        LOG_ERROR("NPU 模型加载失败，请检查 best.rknn 是否推送到开发板同级目录下！");
        return -1;
    }

    // 3. 实例化底层通信，容量设为 3 帧，防视频流卡顿延迟
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(3);

    // 4. 启动视频采集线程，设备号：0
    CameraStreamer streamer(0, frameQueue);
    if (!streamer.start())
    {
        LOG_ERROR("摄像头启动失败！");
        return -1;
    }

    LOG_INFO("主线程：准备将图像给 NPU 进行极速推理...");

    int testFrameCount = 0;
    const int MAX_TEST_FRAMES = 50;

    while(testFrameCount < MAX_TEST_FRAMES)
    {
        cv::Mat currentFrame;

        // 阻塞等待采集线程抓取的新画面（零 CPU 轮询）
        frameQueue.wait_and_pop(currentFrame);

        testFrameCount++;

        // 记录推理前系统时间，用于精准测速
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<DetectResult> results;
        if (inferencer.detect(currentFrame, results))
        {
            // 计算纯 NPU 硬件耗时
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            LOG_INFO("第 {} 帧 NPU 推理完成！硬件耗时 {} ms，发现目标数：{}", testFrameCount, duration.count(), results.size());
        }
    }

    // 5. 关闭系统
    streamer.stop();

    LOG_INFO("======================================");
    LOG_INFO("系统关闭！");
    LOG_INFO("======================================");
    return 0;
}