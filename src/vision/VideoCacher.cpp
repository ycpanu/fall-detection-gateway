#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace vision
    {
        VideoCacher::VideoCacher(int maxFrames) : maxFrame_(maxFrames) {}

        VideoCacher::~VideoCacher()
        {
            // 等待所有后台编码线程写完，避免程序退出时视频文件缺 moov 头而无法播放
            for (auto& t : writerThreads_)
            {
                if (t.joinable())
                {
                    t.join();
                }
            }
        }

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

            // 启动独立后台线程进行耗时视频编码，句柄存入成员供析构时 join
            writerThreads_.emplace_back([snapshot, outputPath, fps]()
            {
                // H.264/MPEG-4 编码器要求宽高为偶数，奇数会写出损坏文件（播放器无法解码）
                int width = snapshot.front().cols;
                int height = snapshot.front().rows;
                width -= width % 2;
                height -= height % 2;
                cv::Size size(width, height);

                cv::VideoWriter writer;

                // 优先 H.264(avc1)：几乎所有现代播放器都原生支持；
                // 精简交叉编译的 OpenCV 若未编译 libx264 会打开失败，则回退 MPEG-4(mp4v)
                writer.open(outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, size);
                if (!writer.isOpened())
                {
                    LOG_WARN("H.264(avc1) 编码器不可用，回退到 MPEG-4(mp4v)...");
                    writer.release();
                    writer.open(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
                }

                if (!writer.isOpened())
                {
                    LOG_ERROR("后台视频编码器打开失败，短视频 {} 生成中断！", outputPath);
                    return ;
                }

                // 将快照中的所有历史帧依次写入视频文件（尺寸不一致时补齐）
                for (const auto& frame : snapshot)
                {
                    if (frame.cols == width && frame.rows == height)
                    {
                        writer.write(frame);
                    }
                    else
                    {
                        cv::Mat resized;
                        cv::resize(frame, resized, size);
                        writer.write(resized);
                    }
                }

                writer.release();
                LOG_WARN("现场视频已成功落盘：{}", outputPath);
            });
        }
    }
}