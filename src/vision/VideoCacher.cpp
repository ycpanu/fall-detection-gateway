#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace vision
    {
        VideoCacher::VideoCacher(int maxFrames) : maxFrame_(maxFrames) {}

        void VideoCacher::pushFrame(const cv::Mat& frame)
        {
            // 加锁保护临界区，防止异步保存时发生数据竞争
            std::lock_guard<std::mutex> lock(mtx_);

            buffer_.push_back(frame.clone());       //存入深拷贝帧

            // 维持滑动窗口大小
            if (buffer_.size() > maxFrame_)
            {
                buffer_.pop_front();
            }
        }

        void VideoCacher::saveVideoAsync(const std::string& outputPath, int fps)
        {
            // 在主线程中快速拷贝出当前的缓存快照，把锁的占用时间降到最低
            std::deque<cv::Mat> snapshot;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                snapshot = buffer_;
            }

            if (snapshot.empty())
            {
                LOG_WARN("视频缓存区为空，无法生成短视频！");
                return ;
            }

            LOG_INFO("已成功截取摔倒前 {} 帧画面，正在后台异步合成视频...", snapshot.size());

            // 启动一个独立的后台线程进行极其耗时的视频编码操作
            std::thread([snapshot, outputPath, fps]()
            {
                // 获取第一帧的尺寸，用于初始化 VideoWriter
                int width = snapshot.front().cols;
                int height = snapshot.front().rows;

                // 使用 mp4v 编码器生成 .mp4 文件
                cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

                if (!writer.isOpened())
                {
                    LOG_ERROR("后台视频编码器打开失败，短视频 {} 生成中断！", outputPath);
                    return ;
                }

                // 将快照中的所有历史帧依次写入视频文件
                for (const auto& frame : snapshot)
                {
                    writer.write(frame);
                }

                writer.release();
                LOG_WARN("现场视频已成功落盘：{}", outputPath);
            }).detach();            //使用 detach 分离线程，让它在后台工作，主线程无需等待
        }
    }
}