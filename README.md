# 基于YOLOv8-Pose与语音关键字识别的多模态跌倒监测系统

> 基于 Orange Pi 3B 的多模态跌倒监测项目，融合 YOLOv8-Pose 视觉检测、离线语音关键词识别、边缘推理、可靠报警传输与云端管理。

## 项目简介

本项目面向居家与室内安全监护场景，设计并实现一套运行在 **Orange Pi 3B（RK3566）** 上的边缘智能跌倒监测系统。

系统在边缘设备本地完成视频采集、人体姿态估计、跌倒判定与语音关键词识别。视觉侧使用 **YOLOv8n-Pose + RKNN Runtime** 获取人体关键点，并结合人体倾角、运动速度和连续帧状态进行跌倒判断；语音侧通过 USB 麦克风运行离线 **Keyword Spotting（KWS）**，识别“救命”“救救我”等紧急求救关键词。

视觉检测与语音识别最终统一进入报警处理流程，并通过 **MQTT、SQLite、FastAPI、Qt** 等模块完成本地报警、断网缓存、网络恢复补传、云端存储和管理端展示。

![alt text](ZFC_1440.jpg)
---

## 系统架构

```mermaid
flowchart LR
    Camera[USB 摄像头] --> Capture[视频采集]
    Capture --> Pose[YOLOv8n-Pose / RKNN]
    Pose --> Tracker[多人目标跟踪]
    Tracker --> Fall[跌倒状态判定]

    Mic[USB 麦克风] --> Audio[ALSA 音频采集]
    Audio --> KWS[离线关键词识别]

    Fall --> Event[统一报警事件]
    KWS --> Event

    Event --> Local[本地报警 / 事件录像]
    Event --> DB[(SQLite)]
    DB --> MQTT[MQTT QoS 1]
    MQTT --> Backend[FastAPI]

    Backend --> Admin[Qt 管理端]
    Backend --> Mini[微信小程序]
```

---

## 核心功能

### 1. 边缘视觉跌倒检测

- 使用 YOLOv8n-Pose 进行人体检测与关键点提取；
- 通过 RKNN Runtime 调用 RK3566 NPU 完成边缘推理；
- 根据躯干倾角、髋部垂直速度、身体中心速度和连续帧状态进行跌倒判断；
- 躯干前倾、框中心下降时，若髋部未快速下降且可信的髋—膝—踝仍保持直立，抑制弯腰造成的疑似跌倒与静态躺卧误报；
- 采用状态机避免单帧姿态误判；
- 支持多人场景，每个目标独立维护跌倒状态；
- 支持安全躺卧区域判断，用于降低床、沙发等场景中的误报警。

典型状态流转：

```text
NORMAL
  ↓
SUSPECTED_FALL
  ↓
CONFIRMED_FALL
  ↓
ALARMED
```

### 2. 离线语音求救识别

- 通过 ALSA 采集 USB 麦克风音频；
- 使用 sherpa-onnx 运行离线 Keyword Spotting；
- 支持“救命”“救救我”等求救关键词；
- 语音识别运行在独立线程中，不依赖云端语音服务；
- 原始音频不长期保存，仅输出关键词识别结果。

### 3. 多模态协同报警

视觉跌倒和语音求救统一转换为报警事件，并记录事件类型、来源、设备、部署区域、时间、人员轨迹及关键词等信息。

当视觉与语音在一定时间窗口内先后触发时，可将两种来源合并到同一个事件中，使管理端看到的是同一次异常的多源证据，而不是两条完全独立的报警。

### 4. 可靠报警与断网容灾

报警链路采用 **Local-First** 思路：

```text
报警产生
   ↓
SQLite PENDING
   ↓
MQTT QoS 1
   ↓
Broker PUBACK
   ↓
SQLite UPLOADED
```

主要设计包括：

- 报警先写入本地 SQLite，再尝试上传；
- MQTT 使用 QoS 1 传输关键报警；
- 网络恢复后由独立补传线程重新发送 `PENDING` 事件；
- 使用唯一 `event_id` 保证重复发送时的幂等性；
- MQTT 支持自动重连和订阅恢复。

### 5. 事件视频与实时监控

- 使用环形缓存保存近期视频帧；
- 视觉跌倒事件触发后异步生成现场短视频；
- 使用 FFmpeg + Rockchip RKMPP 进行 H.264 硬件编码；
- 视频编码任务与 AI 推理线程解耦，避免阻塞实时检测；
- 用户端支持按需开启实时视频，而不是持续上传原始视频。

---

## 多线程设计

边缘端采用 C++17 多线程架构，将实时任务与耗时 I/O 解耦。主要线程包括：

```text
Camera Capture
Audio / KWS
Alert Processing
Retry Upload
Heartbeat
SQLite Worker
Video Encoding
Live Streaming
```

线程之间主要使用 `std::thread`、`std::atomic`、`std::mutex`、`std::condition_variable`、线程安全队列和生产者/消费者模型进行同步与数据交换。

程序支持 `SIGINT / SIGTERM` 优雅退出，并对 MQTT、音频、摄像头及后台工作线程等资源进行统一释放。

---

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 边缘硬件 | Orange Pi 3B / Rockchip RK3566 |
| 操作系统 | Ubuntu 22.04 ARM64 |
| 边缘开发 | C++17 / CMake |
| 姿态估计 | YOLOv8n-Pose |
| NPU 推理 | RKNN Runtime |
| 图像处理 | OpenCV |
| 目标跟踪 | IoU + 中心点距离 |
| 音频采集 | ALSA |
| 关键词识别 | sherpa-onnx KWS |
| 消息通信 | MQTT / Paho MQTT C++ |
| 本地存储 | SQLite3 / WAL |
| 视频处理 | FFmpeg / RKMPP |
| 云端服务 | FastAPI |
| 桌面管理端 | Qt |
| 用户端 | 微信小程序 |
| 日志 | spdlog |
| 数据格式 | JSON / nlohmann-json |

---

## 工程设计重点

### 边缘推理

视觉模型部署在 RK3566 NPU 上运行，将主要感知和判断过程放在本地完成，降低对持续云端计算和视频上传的依赖。

### 多人独立状态

系统不使用单一全局跌倒状态，而是通过轻量级目标跟踪维护 `track_id`，为每个人分别记录姿态历史和跌倒状态。

### 异步任务解耦

视频编码、数据库写盘、报警处理和网络补传均尽量放到后台线程执行，减少对实时视觉推理主流程的阻塞。

### 弱网可靠性

通过 SQLite Local-First、MQTT QoS 1、自动重连、断网补传以及 `event_id` 幂等机制，提高异常网络环境下报警数据的可靠性。

---

## 项目组成

```text
fall-detection-gateway/
├── apps/                 # 边缘端程序入口
├── include/              # C++ 头文件
├── src/                  # 视觉、语音、网络、存储等核心模块
├── configs/              # 系统配置
├── models/               # RKNN / KWS 模型资源
├── cloud_backend/        # FastAPI 后端
├── deploy/               # 部署相关文件
├── CMakeLists.txt
└── README.md
```

配套客户端包括 Qt Desktop Admin 与微信小程序。

---

## 构建与运行

### Pose 模型混合量化

`[1,56,8400]` 的合并输出包含像素坐标以及 `0～1` 的人体、关键点置信度。
若整个输出共用一个 INT8 量化尺度，坐标范围可能导致置信度被量化为零。
转换脚本使用 RKNN Toolkit2 的两步混合量化，默认 `--fp16-scope output`：卷积保持 INT8，
仅将关键点坐标/置信度合并、最终输出合并及其输出别名设为 FP16。
当前模型对应 5 个 FP16 张量，编译结果的 77 个卷积类算子均为 INT8。
旧版将整个检测头设为 FP16，会让大量卷积使用浮点计算；需要复现旧方案时可指定 `--fp16-scope head`。
输入仍由 C++ 提供 RGB uint8，归一化在 RKNN 中完成；C++ 保留 `want_float=1`。

在安装了 RKNN Toolkit2 的环境中，从项目根目录运行：

```bash
python -u models/convert_rknn.py
```

默认读取 `models/yolov8n-pose.onnx` 和 `models/dataset.txt`，输出
`models/yolov8n-pose-hybrid.rknn`。中间文件保存在 `build/rknn_pose_hybrid`。
可用 `--step 1` 和 `--step 2` 分别执行两个阶段；`--calibration-limit 20` 仅用于快速验证流程，
正式转换默认使用完整标定列表。该混合量化路径依据瑞芯微的
[混合量化示例](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/hybrid_quant)。

如果已完成旧版转换，且工作目录中保留了 `.model`、`.data`、`.quantization.cfg`，可复用标定结果：

```bash
python -u models/convert_rknn.py --step 2 --fp16-scope output \
  --output models/yolov8n-pose-hybrid-fast.rknn
```

第二步会重新配置 FP16 范围，清除旧版整个检测头的 FP16 设置，无需再次标定。
若中间文件存放在其他目录，追加 `--work-dir 实际目录`；`scripts/build.sh` 会清理 `build/`，
需要长期保留中间文件时，应选择 `build/` 以外的工作目录。
若中间文件已被删除，去掉 `--step 2` 重新执行完整转换。

启动后会显示 Python 路径、当前阶段和标定图片路径检查进度。
`/mnt/` 下的跨系统文件检查可能较慢；出现 `标定路径检查：...` 时尚未开始模型量化。

将新模型复制到开发板项目的 `models/` 目录，并在 `configs/config.json` 中设置：

```json
"rknn_model_path": "./models/yolov8n-pose-hybrid.rknn"
```

用同一视频复测：启动日志的最终输出应为 FP16，且有人画面的 `[POSE诊断] max_person_conf`
应能恢复非零值。是否达到原检测效果仍以板端视频测试为准。
若使用上述 `hybrid-fast` 文件名，配置中的路径也要相应修改。
模拟器抽样只能验证数值和检测结果，实际 FPS 需在开发板上用同一视频测量，
同时比较 `[PERF]` 的 `rknn_run` 耗时与程序整体 FPS。
程序预热 20 帧后，每 100 帧还会输出 `[PERF] RKNN stages`，分别统计
`preprocess`、`inputs_set`、`run`、`outputs_get`、`postprocess` 的平均耗时。
这些值对应图像预处理、输入提交、模型执行、输出获取、后处理（含输出释放）。
预处理会先缩放 BGR 图像，再转为 RGB 并写入复用的 letterbox 缓冲区。
测试 C++ 优化时继续使用同一个模型和视频，以便比较耗时。

### 512 输入的测速模型

若 640 输入的 `rknn_run` 已接近 100 ms，达到整体 10 FPS 还需要给前后处理留出时间。
可以用同一份权重重新导出 512 输入，保持上述混合量化方式；512 的像素数为 640 的 64%，
实际速度需上板测量，人体和关键点精度也需复测，尤其是远处的小目标。
模型尺寸由导出的计算图决定，不能仅在配置文件中修改图片尺寸。

在安装了 Ultralytics 的导出环境中，从原权重所在目录执行以下 Python 代码。
这里使用原模型对应的 Ultralytics 8.4.116、opset 12，另存权重副本以避免覆盖已有 ONNX：

```python
import shutil
from ultralytics import YOLO

shutil.copy2("yolov8n-pose.pt", "yolov8n-pose-512.pt")
YOLO("yolov8n-pose-512.pt").export(
    format="onnx", imgsz=512, opset=12, simplify=False,
    dynamic=False, nms=False, device="cpu",
)
```

导出参数见 [Ultralytics 官方说明](https://docs.ultralytics.com/modes/export/)。
将生成的 `yolov8n-pose-512.onnx` 放到本项目 `models/`，在 RKNN 转换环境运行：

```bash
python -u models/convert_rknn.py \
  --onnx models/yolov8n-pose-512.onnx \
  --work-dir /tmp/rknn_pose_512 \
  --calibration-limit 20 \
  --output models/yolov8n-pose-512-hybrid-fast-test.rknn
```

这是 20 张标定图的快速验证配置；正式评估请去掉 `--calibration-limit 20`，使用有代表性的完整标定列表。
新尺寸需要重新标定，不能沿用 640 模型的 `.model/.data/.quantization.cfg`。
将模型复制到开发板并修改 `rknn_model_path` 后，启动日志应显示输入 `512×512`、输出 `[1,56,5376]`。
C++ 从模型属性读取输入尺寸与候选数量，无需修改置信度阈值或写死 5376。
先比较同一段视频的 `[PERF]` 日志和跌倒检测结果，再决定是否采用该尺寸。

### 环境要求

边缘端主要依赖：

```text
OpenCV
RKNN Runtime
Paho MQTT C/C++
SQLite3
ALSA
sherpa-onnx / ONNX Runtime
FFmpeg
RKMPP
spdlog
nlohmann-json
```

### 编译

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j$(nproc)
```

运行方式以项目当前 CMake 输出路径为准，例如：

```bash
./bin/fall_detection_gateway
```

系统参数通过配置文件管理，包括摄像头、模型路径、MQTT Broker、设备 ID、部署区域、报警阈值及存储路径等。

---

## 当前进度

目前已完成核心边缘监测链路，包括：

- YOLOv8-Pose RKNN 推理；
- 多人目标跟踪与跌倒状态判定；
- 离线中文 KWS；
- 视觉与语音统一报警处理；
- MQTT 报警上报与自动重连；
- SQLite 本地持久化与断网自动补传；
- 事件视频缓存与硬件编码；
- FastAPI 基础服务；
- Qt 设备与报警管理；
- 小程序报警记录与按需视频功能；
- 边缘程序优雅退出与基础服务化部署。

当前主要工作集中在安全区域可视化配置、微信主动消息通知、视频异常补传以及系统性能与检测效果测试。

---

## 后续测试

项目后续将重点评估：

- 跌倒检测 Precision、Recall、F1-score；
- 正常行为误报情况；
- 单帧推理耗时与实际 FPS；
- 报警端到端响应时间；
- CPU、内存与存储占用；
- MQTT 断网恢复时间；
- 报警补传成功率；
- 不同姿态、距离及场景下的检测稳定性。

---

## 项目亮点

本项目不仅关注模型推理本身，更侧重于 **Linux 环境下完整实时系统的设计与实现**。项目覆盖了：

- ARM Linux 边缘 AI 部署；
- C++ 多线程与生产者/消费者模型；
- USB 摄像头与 ALSA 音频采集；
- NPU 推理与硬件视频编码；
- MQTT 网络通信与弱网容灾；
- SQLite 本地可靠存储；
- GDB 多线程故障定位；
- Systemd 服务化运行；
- FastAPI、Qt 与小程序多端协同。

---

