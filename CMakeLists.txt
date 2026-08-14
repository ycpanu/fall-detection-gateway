cmake_minimum_required(VERSION 3.14)
project(FallDetectionGateway)

#设置 C++ 标准为 17 ，paho-mqtt-cpp 依赖 std::string_view/std::variant 等 C++17 特性
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
#输出目录：可执行文件 -> bin，动态链接库 -> build/lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#指定生成可执行文件
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
add_executable(fall_detection_gateway apps/main.cpp ${SRC_FILES})

#链接核心动态库
target_link_libraries(fall_detection_gateway
    PRIVATE
    spdlog::spdlog
    pthread
    ${OpenCV_LIBS}
    ${RKNN_LIB}
    rknnrt
    paho-mqttpp3
    paho-mqtt3a
    Threads::Threads
)

# 允许共享库中存在未解析符号——这些符号在目标设备运行时由设备上的系统库提供
target_link_options(fall_detection_gateway PRIVATE -Wl,--allow-shlib-undefined)
