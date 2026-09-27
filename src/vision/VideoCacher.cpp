#include <opencv2/videoio.hpp>
#include <cstdlib>
#include <cstdio>
#include <filesystem>

#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/utils/SysLogger.hpp"
#include "fall-detection/utils/ConfigManager.hpp"

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
            std::vector<VideoTask> completedTasks;

            {
                std::lock_guard<std::mutex> lock(bufferMtx_);

                cv::Mat cachedFrame = frame.clone();
                buffer_.push_back(cachedFrame);

                if (buffer_.size() > maxFrame_)
                    buffer_.pop_front();

                for (auto it = pendingRecords_.begin(); it != pendingRecords_.end();)
                {
                    it->frames.push_back(cachedFrame);
                    --it->remainingPostFrames;

                    if (it->remainingPostFrames <= 0)
                    {
                        completedTasks.push_back({
                            std::move(it->frames),
                            it->outputPath,
                            it->fps
                        });

                        it = pendingRecords_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }

            if (!completedTasks.empty())
            {
                {
                    std::lock_guard<std::mutex> lock(queueMtx_);

                    for (auto& task : completedTasks)
                        taskQueue_.push(std::move(task));
                }

                cv_.notify_one();
            }
        }

        void VideoCacher::saveVideoAsync(const std::string& outputPath, int fps, int postFrames)
        {
            std::deque<cv::Mat> snapshot;

            {
                std::lock_guard<std::mutex> lock(bufferMtx_);
                snapshot = buffer_;

                if (snapshot.empty())
                {
                    LOG_WARN("视频缓存区为空，无法生成事件视频！");
                    return;
                }

                if (postFrames > 0)
                {
                    PendingVideoRecord record;
                    record.frames = std::move(snapshot);
                    record.outputPath = outputPath;
                    record.fps = fps;
                    record.remainingPostFrames = postFrames;

                    pendingRecords_.push_back(std::move(record));

                    LOG_INFO("已锁定报警前 {} 帧，继续采集报警后 {} 帧", buffer_.size(), postFrames);
                    return;
                }
            }

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

                std::filesystem::path outputPath(task.outputPath);

                if (outputPath.has_parent_path())
                {
                    std::filesystem::create_directories(outputPath.parent_path());
                }

                std::string command =
                    "ffmpeg -y -loglevel error "
                    "-f rawvideo "
                    "-pix_fmt bgr24 "
                    "-s " + std::to_string(width) + "x" + std::to_string(height) + " "
                    "-r " + std::to_string(task.fps) + " "
                    "-i - "
                    "-an "
                    "-vf format=nv12 "
                    "-c:v h264_rkmpp "
                    "-movflags +faststart "
                    "\"" + task.outputPath + "\"";

                FILE* pipe = popen(command.c_str(), "w");

                if (!pipe)
                {
                    LOG_ERROR("无法启动 FFmpeg 硬件编码器：{}", task.outputPath);
                    continue;
                }

                bool writeOk = true;

                for (const auto& frame : task.frames)
                {
                    cv::Mat outputFrame;

                    if (frame.cols != width || frame.rows != height)
                        cv::resize(frame, outputFrame, size);
                    else
                        outputFrame = frame;

                    if (!outputFrame.isContinuous())
                        outputFrame = outputFrame.clone();

                    size_t frameBytes = outputFrame.total() * outputFrame.elemSize();

                    size_t written = fwrite(
                        outputFrame.data,
                        1,
                        frameBytes,
                        pipe
                    );

                    if (written != frameBytes)
                    {
                        LOG_ERROR("向 FFmpeg 写入视频帧失败");
                        writeOk = false;
                        break;
                    }
                }

                int ffmpegRet = pclose(pipe);

                if (!writeOk || ffmpegRet != 0)
                {
                    LOG_ERROR("FFmpeg 硬件编码失败：{}", task.outputPath);
                    continue;
                }

                LOG_INFO("事件视频硬件编码完成：{}", task.outputPath);

                // 利用 curl 后台上传视频到云端
                std::string serverUrl = utils::ConfigManager::getInstance().getString("network.api_base_url", "http://10.48.212.22:8000");
                // 构建上传命令：curl -s -X POST -F "file=@videos/fall_xxx.mp4" http://ip:8000/api/upload/video
                std::string uploadCmd = "curl -s -X POST -F \"file=@" + task.outputPath + "\" " + serverUrl + "/api/upload/video";
                
                LOG_INFO("正在后台上传短视频到云端: {}", task.outputPath);
                int uploadRet = std::system(uploadCmd.c_str());
                if (uploadRet == 0)
                {
                    LOG_INFO("现场短视频上传云端成功！");
                }
                else
                {
                    LOG_ERROR("短视频上传失败！可能网络断开，视频已保留在本地。");
                }
            }
        }
    }
}