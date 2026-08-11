#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/concurrency/ThreadSafeQueue.hpp"
#include "fall-detection/vision/CameraStreamer.hpp"

using namespace fall_detection;

int main(int argc, char** argv)
{
    utils::SysLogger::getInstance().init("logs/gateway.log");

    LOG_INFO("===========================================");
    LOG_INFO("边缘网关系统启动");
    LOG_INFO("===========================================");

    // 2. 实例化底层通信
    // 设定队列最大容量为 3
    concurrency::ThreadSafeQueue<cv::Mat> frameQueue(3);

    // 3. 实例化并启动视频采集线程
    CameraStreamer streamer(0, frameQueue);
    if (!streamer.start())
    {
        LOG_ERROR("摄像头启动失败！请检查 USB 摄像头是否正确连接到开发板");
        return -1;
    }

    LOG_INFO("主线程：准备从队列中获取图像...");

    // 4. 模拟 AI 推理主循环
    int testFrameCount = 0;
    const int MAX_TEST_FRAMES = 50; //测试拉取 50 帧后自动安全退出

    while (testFrameCount < MAX_TEST_FRAMES)
    {
        cv::Mat currentFrame;

        // 阻塞等待，直到采集线程 push 新画面唤醒它，实现零 CPU 轮询空转
        frameQueue.wait_and_pop(currentFrame);

        if (!currentFrame.empty())
        {
            testFrameCount++;
            LOG_INFO("成功获取第{}帧，分辨率：{}x{}，通道数：{}", testFrameCount, currentFrame.cols, currentFrame.rows, currentFrame.channels());

            // 把第 10 帧真正保存到磁盘上，证明摄像头确实抓到了画面
            if (testFrameCount == 10)
            {
                std::string savePath = "test_capture.jpg";
                cv::imwrite(savePath, currentFrame);
                LOG_INFO(">>> 已将第 10 帧画面保存至当权目录的{}，请在测试结束后查看。<<<",savePath);
            }
        }

        // 模拟边缘 AI 推理硬件加速推理的耗时（假设 NPU 推理需要 100 毫秒）
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }

    // 5. 关闭系统，释放硬件资源
    LOG_INFO("测试帧数达到设定值，准备执行关闭");
    streamer.stop();

    LOG_INFO("==========================================");
    LOG_INFO("关闭系统");
    LOG_INFO("==========================================");

    return 0;
}