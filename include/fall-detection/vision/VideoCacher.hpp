#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fall_detection
{
    namespace vision
    {
        /**
         * @brief 摔倒短视频异步缓存器（Ring Buffer 架构）
         * 
         * 负责在内存中实时维持一个固定长度的短视频滑动窗口
         * 当报警触发时，利用异步线程将缓存帧快速持久化为 MP4 视频文件
         */
        class VideoCacher
        {
            public:
                /**
                 * @brief 构造函数
                 * @param maxFrames 内存中保留的最大历史帧数
                 * 
                 */
                VideoCacher(int maxFrames = 90);
                ~VideoCacher();

                /**
                 * @brief 将最新的一帧画面压入环形缓存
                 * @param frame 当前摄像头画面
                 */
                void pushFrame(const cv::Mat& frame);

                /**
                 * @brief 异步保存当前缓存的视频片段（不阻塞主线程）
                 * @param outputPath 视频保存路径
                 * @param fps 视频保存帧率
                 */
                void saveVideoAsync(const std::string& outputPath, int fps = 30);
            
            private:
                int maxFrame_;
                std::deque<cv::Mat> buffer_;
                std::mutex mtx_;
                std::vector<std::thread> writerThreads_;   // 后台编码线程句柄，析构时统一 join 保证视频写完
        };
    }
}