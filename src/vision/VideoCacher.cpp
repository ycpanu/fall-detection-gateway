#include "fall-detection/vision/VideoCacher.hpp"
#include "fall-detection/utils/SysLogger.hpp"

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
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
            stopAndDrain();
        }

        void VideoCacher::stopAndDrain()
        {
            {
                std::scoped_lock lock(bufferMtx_, queueMtx_);
                if (!isRunning_.exchange(false))
                    return;
                for (auto& record : pendingRecords_)
                {
                    LOG_WARN(
                        "[VideoCacher] 采集已停止，使用已有帧保存视频：event_id={}, missing_post_frames={}",
                        record.eventId, record.remainingPostFrames);
                    taskQueue_.push({std::move(record.frames),
                        std::move(record.eventId), std::move(record.outputPath), record.fps});
                }
                pendingRecords_.clear();
            }
            cv_.notify_all();
            if (workerThread_.joinable())
                workerThread_.join();
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

                // FFmpeg 使用独立进程组：终端 Ctrl+C 只通知网关，编码得以收尾。
                // 先写临时文件，成功退出后再发布完整的 MP4。
                auto temporaryPath = outputPath;
                temporaryPath += ".part.mp4";
                const std::string sizeArg =
                    std::to_string(width) + "x" + std::to_string(height);
                const std::string fpsArg = std::to_string(task.fps);
                std::string temporaryName = temporaryPath.string();
                char* args[] = {
                    const_cast<char*>("ffmpeg"), const_cast<char*>("-y"),
                    const_cast<char*>("-loglevel"), const_cast<char*>("error"),
                    const_cast<char*>("-f"), const_cast<char*>("rawvideo"),
                    const_cast<char*>("-pix_fmt"), const_cast<char*>("bgr24"),
                    const_cast<char*>("-s"), const_cast<char*>(sizeArg.c_str()),
                    const_cast<char*>("-r"), const_cast<char*>(fpsArg.c_str()),
                    const_cast<char*>("-i"), const_cast<char*>("-"),
                    const_cast<char*>("-an"), const_cast<char*>("-vf"),
                    const_cast<char*>("format=nv12"), const_cast<char*>("-c:v"),
                    const_cast<char*>("h264_rkmpp"), const_cast<char*>("-movflags"),
                    const_cast<char*>("+faststart"), temporaryName.data(), nullptr
                };
                int pipeFds[2];
                if (::pipe(pipeFds) != 0)
                {
                    LOG_ERROR(
                        "[VideoCacher] 创建 FFmpeg 输入管道失败：event_id={}, errno={}",
                        task.eventId, errno
                    );
                    continue;
                }
                posix_spawn_file_actions_t actions;
                posix_spawnattr_t attributes;
                posix_spawn_file_actions_init(&actions);
                posix_spawn_file_actions_adddup2(&actions, pipeFds[0], STDIN_FILENO);
                posix_spawn_file_actions_addclose(&actions, pipeFds[0]);
                posix_spawn_file_actions_addclose(&actions, pipeFds[1]);
                posix_spawnattr_init(&attributes);
                posix_spawnattr_setpgroup(&attributes, 0);
                posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
                pid_t child = -1;
                const int spawnResult = posix_spawnp(
                    &child, "ffmpeg", &actions, &attributes, args, environ);
                posix_spawnattr_destroy(&attributes);
                posix_spawn_file_actions_destroy(&actions);
                ::close(pipeFds[0]);
                if (spawnResult != 0)
                {
                    ::close(pipeFds[1]);
                    LOG_ERROR("[VideoCacher] 启动 FFmpeg 失败：event_id={}, error={}",
                        task.eventId, spawnResult);
                    continue;
                }
                FILE* pipe = ::fdopen(pipeFds[1], "w");
                if (!pipe)
                {
                    ::close(pipeFds[1]);
                    int status = 0;
                    while (::waitpid(child, &status, 0) == -1 && errno == EINTR) {}
                    LOG_ERROR("[VideoCacher] 打开 FFmpeg 输入流失败：event_id={}", task.eventId);
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

                if (::fclose(pipe) != 0)
                    writeOk = false;
                int ffmpegStatus = 0;
                int waited;
                do {
                    waited = ::waitpid(child, &ffmpegStatus, 0);
                } while (waited == -1 && errno == EINTR);

                if (!writeOk ||
                    waited != child || !WIFEXITED(ffmpegStatus) ||
                    WEXITSTATUS(ffmpegStatus) != 0)
                {
                    std::error_code removeError;
                    std::filesystem::remove(temporaryPath, removeError);
                    LOG_ERROR(
                        "[VideoCacher] 视频硬件编码失败：event_id={}, path={}, status={}",
                        task.eventId,
                        task.outputPath, ffmpegStatus
                    );

                    continue;
                }

                std::error_code renameError;
                std::filesystem::rename(temporaryPath, outputPath, renameError);
                if (renameError)
                {
                    LOG_ERROR("[VideoCacher] 视频文件落盘失败：event_id={}, error={}",
                        task.eventId, renameError.message());
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
