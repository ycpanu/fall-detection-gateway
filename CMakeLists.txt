cmake_minimum_required(VERSION 3.10)
project(FallDetectionGateway)

#设置 C++ 标准为 14 ，适应后续 OpenCV 和多线程需求
set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sysroot/usr/lib/aarch64-linux-gnu/cmake/opencv4")

#寻找 Sysroot 里的 OpenCV 库
find_package(OpenCV REQUIRED)

find_package(spdlog REQUIRED)
#包含 OpenCV 头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

#指定生成可执行文件
add_executable(fall_gateway apps/main.cpp)

#链接核心动态库
target_link_libraries(fall_gateway 
    PRIVATE 
    spdlog::spdlog 
    pthread
    ${OpenCV_LIBS} 
)
