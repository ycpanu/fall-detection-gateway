#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <cstdio>
#include <filesystem>
#include <utility>

namespace fall_detection
{
    namespace vision
    {

        VideoCacher::VideoCacher(int maxFrames)
            : maxFrame_(maxFrames),
            isRunning_(true)
        {
            workerThread_ = std::thread(
                &VideoCacher::encodingWorkerLoop,
                this
            );
        }

        VideoCacher::~VideoCacher()
        {
            isRunning_ = false;
            cv_.notify_all();

            if (workerThread_.joinable())
            {
                workerThread_.join();
            }
        }

        void VideoCacher::setVideoReadyCallback(
            VideoReadyCallback callback)
        {
            videoReadyCallback_ = std::move(callback);
        }

        void VideoCacher::pushFrame(const cv::Mat& frame)
        {
            std::vector<VideoTask> completedTasks;

            {
                std::lock_guard<std::mutex> lock(bufferMtx_);

                cv::Mat cachedFrame = frame.clone();
                buffer_.push_back(cachedFrame);

                if (buffer_.size() > static_cast<std::size_t>(maxFrame_))
                {
                    buffer_.pop_front();
                }

                for (auto it = pendingRecords_.begin();
                    it != pendingRecords_.end();)
                {
                    it->frames.push_back(cachedFrame);
                    --it->remainingPostFrames;

                    if (it->remainingPostFrames <= 0)
                    {
                        VideoTask task;

                        task.frames = std::move(it->frames);
                        task.eventId = it->eventId;
                        task.outputPath = it->outputPath;
                        task.fps = it->fps;

                        completedTasks.push_back(
                            std::move(task)
                        );

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
                    {
                        taskQueue_.push(
                            std::move(task)
                        );
                    }
                }

                cv_.notify_one();
            }
        }

        void VideoCacher::saveVideoAsync(
            const std::string& eventId,
            const std::string& outputPath,
            int fps,
            int postFrames)
        {
            std::deque<cv::Mat> snapshot;

            {
                std::lock_guard<std::mutex> lock(bufferMtx_);

                snapshot = buffer_;

                if (snapshot.empty())
                {
                    LOG_WARN(
                        "[VideoCacher] 视频缓存为空，无法生成事件视频：event_id={}",
                        eventId
                    );

                    return;
                }

                if (postFrames > 0)
                {
                    PendingVideoRecord record;

                    record.frames = std::move(snapshot);
                    record.eventId = eventId;
                    record.outputPath = outputPath;
                    record.fps = fps;
                    record.remainingPostFrames = postFrames;

                    pendingRecords_.push_back(
                        std::move(record)
                    );

                    LOG_INFO(
                        "[VideoCacher] 已锁定报警前 {} 帧，继续采集报警后 {} 帧：event_id={}",
                        buffer_.size(),
                        postFrames,
                        eventId
                    );

                    return;
                }
            }

            VideoTask task;

            task.frames = std::move(snapshot);
            task.eventId = eventId;
            task.outputPath = outputPath;
            task.fps = fps;

            {
                std::lock_guard<std::mutex> lock(queueMtx_);

                taskQueue_.push(
                    std::move(task)
                );
            }

            cv_.notify_one();
        }

        void VideoCacher::encodingWorkerLoop()
        {
            while (true)
            {
                VideoTask task;

                {
                    std::unique_lock<std::mutex> lock(queueMtx_);

                    cv_.wait(
                        lock,
                        [this]()
                        {
                            return !taskQueue_.empty() ||
                                !isRunning_;
                        }
                    );

                    if (!isRunning_ && taskQueue_.empty())
                    {
                        break;
                    }

                    task = std::move(
                        taskQueue_.front()
                    );

                    taskQueue_.pop();
                }

                if (task.frames.empty())
                {
                    LOG_WARN(
                        "[VideoCacher] 编码任务没有视频帧：event_id={}",
                        task.eventId
                    );

                    continue;
                }

                int width = task.frames.front().cols;
                int height = task.frames.front().rows;

                width -= width % 2;
                height -= height % 2;

                if (width <= 0 || height <= 0)
                {
                    LOG_ERROR(
                        "[VideoCacher] 视频尺寸无效：event_id={}, width={}, height={}",
                        task.eventId,
                        width,
                        height
                    );

                    continue;
                }

                const cv::Size outputSize(
                    width,
                    height
                );

                std::filesystem::path outputPath(
                    task.outputPath
                );

                if (outputPath.has_parent_path())
                {
                    std::error_code ec;

                    std::filesystem::create_directories(
                        outputPath.parent_path(),
                        ec
                    );

                    if (ec)
                    {
                        LOG_ERROR(
                            "[VideoCacher] 创建视频目录失败：path={}, error={}",
                            outputPath.parent_path().string(),
                            ec.message()
                        );

                        continue;
                    }
                }

                const std::string command =
                    "ffmpeg -y -loglevel error "
                    "-f rawvideo "
                    "-pix_fmt bgr24 "
                    "-s " +
                    std::to_string(width) +
                    "x" +
                    std::to_string(height) +
                    " "
                    "-r " +
                    std::to_string(task.fps) +
                    " "
                    "-i - "
                    "-an "
                    "-vf format=nv12 "
                    "-c:v h264_rkmpp "
                    "-movflags +faststart "
                    "\"" +
                    task.outputPath +
                    "\"";

                FILE* pipe = popen(
                    command.c_str(),
                    "w"
                );

                if (!pipe)
                {
                    LOG_ERROR(
                        "[VideoCacher] 无法启动 FFmpeg：event_id={}, path={}",
                        task.eventId,
                        task.outputPath
                    );

                    continue;
                }

                bool writeOk = true;

                for (const auto& frame : task.frames)
                {
                    cv::Mat outputFrame;

                    if (frame.cols != width ||
                        frame.rows != height)
                    {
                        cv::resize(
                            frame,
                            outputFrame,
                            outputSize
                        );
                    }
                    else
                    {
                        outputFrame = frame;
                    }

                    if (!outputFrame.isContinuous())
                    {
                        outputFrame =
                            outputFrame.clone();
                    }

                    const std::size_t frameBytes =
                        outputFrame.total() *
                        outputFrame.elemSize();

                    const std::size_t written =
                        fwrite(
                            outputFrame.data,
                            1,
                            frameBytes,
                            pipe
                        );

                    if (written != frameBytes)
                    {
                        LOG_ERROR(
                            "[VideoCacher] 向 FFmpeg 写入视频帧失败：event_id={}",
                            task.eventId
                        );

                        writeOk = false;
                        break;
                    }
                }

                const int ffmpegRet =
                    pclose(pipe);

                if (!writeOk ||
                    ffmpegRet != 0)
                {
                    LOG_ERROR(
                        "[VideoCacher] 视频硬件编码失败：event_id={}, path={}",
                        task.eventId,
                        task.outputPath
                    );

                    continue;
                }

                LOG_INFO(
                    "[VideoCacher] 事件视频编码完成：event_id={}, path={}",
                    task.eventId,
                    task.outputPath
                );

                if (videoReadyCallback_)
                {
                    videoReadyCallback_(
                        task.eventId,
                        task.outputPath
                    );
                }
            }
        }

    }
}