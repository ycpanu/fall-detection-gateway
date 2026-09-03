#!/bin/bash
# 开启严格模式：遇到任何错误立即退出，未定义的变量报错
set -euo pipefail

# 1. 强制锚定项目根目录 (无论在哪里执行该脚本，都能精准切回项目根目录)
# 解析当前脚本的绝对路径，并向上推一级获取根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "开始交叉编译边缘网关 (Fall Detection Gateway)"
echo "工作目录已锁定: $PROJECT_ROOT"
echo "=================================================="

# 2. 安全构建目录管理
mkdir -p build
cd build

# 核弹级清理缓存，确保 CMake 强制读取 aarch64-toolchain.cmake
echo "正在清理构建缓存..."
rm -rf *

# 3. 交叉编译配置
echo "正在生成构建配置..."
cmake .. -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake

# 4. 极速多核编译
echo "正在启动多核编译..."
make -j$(nproc)

echo "=================================================="
echo "编译大功告成！"
echo "产物已输出至: $PROJECT_ROOT/bin/fall_detection_gateway"
echo "=================================================="