#include "fall-detection/utils/SysLogger.hpp"

#include <vector>
#include <iostream>

namespace fall_detection
{
    namespace utils
    {
        void SysLogger::init(const std::string& log_file_path)
        {
            try
            {
                {
                    // 1. 初始化异步日志线程池
                    // 队列大小为 8192，拥有 1 个后台线程专门负责写磁盘
                    // 核心价值：这样写日志不会阻塞后续的 AI 推理和摄像头抓图主线程
                    spdlog::init_thread_pool(8192, 1);

                    // 2. 创建控制台输出目标
                    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

                    // 3. 创建文件输出目标
                    // 参数：保存路径，单个文件最大 5MB，最大保留 3 个滚动文件
                    // 防止设备长期运行，日志文件无限大把 SD 卡占满
                    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_file_path, 1024 * 1024 * 5, 3);

                    // 4. 将两个输出目标组合起来
                    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};

                    // 5. 创建全局异步日志器 "GatewayLogger"
                    logger_ = std::make_shared<spdlog::async_logger>(
                        "GatewayLogger",
                        sinks.begin(),
                        sinks.end(),
                        spdlog::thread_pool(),
                        spdlog::async_overflow_policy::block //如果写太快导致队列满了，阻塞一下防止丢失致命报错
                    );

                    // 6. 设置日志格式
                    // 格式说明：[时间] [线程ID] [日志级别] [源文件:行号] 具体的日志信息
                    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] [%s:%#] %v");

                    // 7. 设置最低生效的日志级别为 DEBUG
                    logger_->set_level(spdlog::level::debug);

                    // 8. 容灾设置：当遇到 ERROR 级别的日志时，立刻强制将内存中的日志刷入磁盘
                    // 防止程序遇到段错误（Segfault）死机重启时，还没来得及写盘，导致死机现场丢失
                    logger_->flush_on(spdlog::level::err);

                    // 9. 注册 spdlog 的默认 Logger，并在程序退出时自动接管清理
                    spdlog::register_logger(logger_);

                }
            }
            catch(const spdlog::spdlog_ex& ex)
            {
                // 如果日志系统本身初始化失败，用 cout 打印
                std::cout << "SysLogger initialization failed: " << ex.what() << std::endl;
            }
            
        }
    }
}