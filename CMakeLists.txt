cmake_minimum_required(VERSION 3.14)
project(FallDetectionGateway)

#设置 C++ 标准为 14 ，适应后续 OpenCV 和多线程需求
set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sysroot/usr/lib/aarch64-linux-gnu/cmake/opencv4")

#寻找 Sysroot 里的 OpenCV 库
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# 通过源码编译获取 fmt 和 spdlog，避免依赖 sysroot 中缺失的 .so 文件
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.0
)
FetchContent_MakeAvailable(fmt)

set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

#包含 OpenCV 和项目头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/sysroot/usr/include
    )
# 动态库
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/sysroot/usr/lib
)
#指定生成可执行文件
add_executable(fall_gateway 
    apps/main.cpp
    src/utils/SysLogger.cpp
    src/vision/CameraStreamer.cpp
    src/vision/RKNNInferencer.cpp
    )

#链接核心动态库
target_link_libraries(fall_gateway
    PRIVATE
    spdlog::spdlog
    pthread
    ${OpenCV_LIBS}
    rknnrt
    Threads::Threads
)

# 允许共享库中存在未解析符号——这些符号在目标设备运行时由设备上的系统库提供
target_link_options(fall_gateway PRIVATE -Wl,--allow-shlib-undefined)
