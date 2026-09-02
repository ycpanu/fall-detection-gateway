#include <opencv2/videoio.hpp>

#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/utils/SysLogger.hpp"

namespace fall_detection
{
    namespace vision
    {
        VideoCacher::VideoCacher(int maxFrames) 
            : maxFrame_(maxFrames), isRunning_(true)
        {
            // 启动单例常驻编码线程
            workerThread_ = std::thread(&VideoCacher::encodingWorkerLoop, this);
        }

        VideoCacher::~VideoCacher()
        {
            // 优雅停止并等待后台编码线程写完最后一个视频
            isRunning_ = false;
            cv_.notify_one();
            if (workerThread_.joinable())
            {
                workerThread_.join();
            }
        }

        void VideoCacher::pushFrame(const cv::Mat& frame)
        {
            std::lock_guard<std::mutex> lock(bufferMtx_);
            buffer_.push_back(frame.clone());

            if (buffer_.size() > maxFrame_)
            {
                buffer_.pop_front();
            }
        }

        void VideoCacher::saveVideoAsync(const std::string& outputPath, int fps)
        {
            // 1. 极速拷贝快照，最小化主存锁占用时间
            std::deque<cv::Mat> snapshot;
            {
                std::lock_guard<std::mutex> lock(bufferMtx_);
                snapshot = buffer_;
            }

            if (snapshot.empty())
            {
                LOG_WARN("视频缓存区为空，无法生成短视频！");
                return;
            }

            LOG_INFO("已成功截取摔倒前 {} 帧画面，投递至后台编码队列...", snapshot.size());

            // 2. 将快照推入编码任务队列并唤醒消费者
            {
                std::lock_guard<std::mutex> lock(queueMtx_);
                taskQueue_.push({std::move(snapshot), outputPath, fps});
            }
            cv_.notify_one();
        }

        void VideoCacher::encodingWorkerLoop()
        {
            while (isRunning_)
            {
                VideoTask task;
                
                // 3. 阻塞等待编码任务，零 CPU 消耗
                {
                    std::unique_lock<std::mutex> lock(queueMtx_);
                    cv_.wait(lock, [this]() { return !taskQueue_.empty() || !isRunning_; });

                    if (!isRunning_ && taskQueue_.empty())
                    {
                        break;
                    }

                    task = std::move(taskQueue_.front());
                    taskQueue_.pop();
                }

                // 4. 耗时编码操作在锁外执行，绝不阻塞主摄像头抓图流水线
                int width = task.frames.front().cols;
                int height = task.frames.front().rows;
                width -= width % 2;
                height -= height % 2;
                cv::Size size(width, height);

                cv::VideoWriter writer;
                writer.open(task.outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), task.fps, size);
                
                if (!writer.isOpened())
                {
                    LOG_WARN("H.264 编码器不可用，回退到 MPEG-4(mp4v)...");
                    writer.release();
                    writer.open(task.outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), task.fps, size);
                }

                if (!writer.isOpened())
                {
                    LOG_ERROR("后台视频编码器打开失败，短视频 {} 生成中断！", task.outputPath);
                    continue;
                }

                for (const auto& frame : task.frames)
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
                LOG_WARN("现场短视频已成功异步落盘：{}", task.outputPath);
            }
        }
    }
}