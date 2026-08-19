# 摔倒检测边缘网关（Fall Detection Gateway）

基于 Rockchip NPU 的摔倒检测边缘网关，运行在 OrangePi 等 aarch64 开发板上。系统通过摄像头实时采集画面，利用 YOLOv8n-Pose 模型在 NPU 上做人体骨骼关键点推理，再由规则引擎判断是否发生摔倒；确认摔倒后触发本地蜂鸣器报警，并将报警事件与现场短视频上传至云端（MQTT），断网时自动落盘、恢复后补传。

## 模块架构

系统由以下六个模块在 `main.cpp` 中融合组装，通过线程安全队列解耦：

| 模块 | 职责 |
| --- | --- |
| `CameraStreamer` | 摄像头 / 本地视频采集，独立线程抓帧 |
| `VideoCacher` | 环形缓存滑窗，摔倒时异步合成前 N 秒短视频 |
| `RKNNInferencer` | NPU 硬件推理，解码 17 个骨骼关键点 + NMS |
| `FallRuleEngine` | 身体轴线角度 + 髋部下坠速度 + 时序状态机判定摔倒 |
| `MqttClient` | 与云端 MQTT Broker 长连接，发布报警 JSON |
| `LocalDatabase` | SQLite 本地容灾，断网积压、恢复补传（Store-and-Forward） |
| `BuzzerController` | 通过 Sysfs 操控 GPIO 触发蜂鸣器 |

数据流：`CameraStreamer → 帧队列 → RKNNInferencer → FallRuleEngine → 报警队列 → MqttClient / LocalDatabase`

## 目录结构

```
fall-detection-gateway/
├── apps/           # 入口 main.cpp
├── include/        # 头文件（fall-detection/{vision,network,hardware,concurrency,utils}）
├── src/            # 各模块实现
├── configs/        # 运行时配置文件 config.json
├── models/         # 模型文件（*.rknn）
├── sysroot/        # 交叉编译依赖 sysroot
├── lib/            # 第三方库（rknpu2 / rknn-toolkit2）
├── tests/          # 测试
├── benchmarks/     # 基准测试
├── tools/          # 辅助脚本
├── aarch64-toolchain.cmake
└── CMakeLists.txt
```

## 编译构建

交叉编译到 aarch64（需先安装 `aarch64-linux-gnu-gcc/g++`）：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake
make -j$(nproc)
```

产物输出到 `bin/fall_detection_gateway`。`fmt`、`spdlog` 通过 FetchContent 源码拉取编译，`rknnrt`、`paho-mqtt`、`sqlite3`、`OpenCV` 来自 `sysroot/`。

## 运行

```bash
# 使用摄像头（设备号由 config.json 配置）
./fall_detection_gateway

# 回放本地视频（用于回归测试）
./fall_detection_gateway /path/to/test.mp4
```

> 运行前请确认 `configs/config.json` 与模型文件（`./yolov8n-pose.rknn`）就位。

---

# 配置系统

系统启动时由单例 `ConfigManager` 加载 `configs/config.json`，并向各模块分发参数。**所有字段均有内置默认值**：配置文件缺失、无法打开或某字段缺失时，系统按默认值继续运行，不会启动失败。

## config.json 字段说明

### system — 系统

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `log_file_path` | string | `logs/gateway.log` | 日志文件路径（相对路径基于可执行文件目录解析） |
| `log_level` | string | `INFO` | 日志最低级别：`TRACE/DEBUG/INFO/WARN/ERROR/CRITICAL`（不区分大小写） |

### vision — 视觉采集

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `camera_device_id` | int | `0` | 摄像头设备号（`0` 即 `/dev/video0`） |
| `frame_queue_size` | int | `3` | 图像帧队列最大容量 |
| `video_cache_frames` | int | `90` | 内存滑动窗口保留的历史帧数 |
| `video_save_fps` | int | `30` | 短视频落盘帧率 |
| `video_output_dir` | string | `videos` | 现场短视频输出目录（相对路径基于可执行文件目录） |

### model — 模型推理

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `rknn_model_path` | string | `./yolov8n-pose.rknn` | RKNN 模型文件路径 |
| `confidence_threshold` | float | `0.5` | 目标置信度过滤阈值 |
| `nms_threshold` | float | `0.45` | NMS 交并比剔除阈值 |

### rule_engine — 摔倒判定规则引擎

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `kpt_conf_threshold` | float | `0.3` | 关键点置信度阈值（低于视为不可见） |
| `fall_angle_threshold` | float | `60.0` | 身体轴线与垂直方向夹角阈值（度） |
| `fall_velocity_threshold` | float | `400.0` | 髋部下坠速度阈值（像素/秒） |
| `confirm_frames_threshold` | int | `5` | 下坠后需连续躺倒的报警帧数（动态摔倒） |
| `static_lie_threshold` | int | `30` | 无下坠时持续躺倒的兜底报警帧数（静态摔倒） |
| `fall_event_window` | int | `15` | 快速下坠事件的有效窗口（帧） |

### network — 网络通信

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mqtt_broker` | string | `tcp://broker.emqx.io:1883` | MQTT Broker 地址 |
| `mqtt_client_id` | string | `OrangePi_Gateway_001` | 网关唯一设备标识 |
| `alert_topic` | string | `fall_detection/alerts` | 报警发布主题 |
| `keep_alive_seconds` | int | `20` | MQTT 心跳保活间隔（秒） |
| `alert_queue_size` | int | `10` | 报警事件队列最大容量 |

### hardware — 硬件

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `buzzer_gpio_pin` | int | `138` | 蜂鸣器 GPIO 引脚号 |
| `alarm_duration_ms` | int | `3000` | 蜂鸣器报警时长（毫秒） |

### storage — 存储

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sqlite_db_path` | string | `fall_detection.db` | SQLite 容灾数据库文件路径 |

## ConfigManager 使用

`ConfigManager` 采用单例模式，头文件 `include/fall-detection/utils/ConfigManager.hpp`：

```cpp
#include "fall-detection/utils/ConfigManager.hpp"

// 1. 启动时加载一次
auto& config = fall_detection::utils::ConfigManager::getInstance();
if (!config.load("configs/config.json")) {
    // 加载失败，后续 getter 均返回默认值
}

// 2. 按模块命名 getter 读取（推荐，类型安全、语义清晰）
int gpio = config.getBuzzerGpioPin();
std::string broker = config.getMqttBroker();
float angle = config.getFallAngleThreshold();

// 3. 或使用通用点号取值接口
int n = config.getInt("vision.frame_queue_size", 3);
std::string s = config.getString("network.alert_topic", "fall_detection/alerts");
double d = config.getDouble("model.confidence_threshold", 0.5);
bool b = config.getBool("some.flag", false);
```

### 新增一个配置项

1. 在 `configs/config.json` 对应分组下添加字段；
2. 在 `ConfigManager.hpp` 声明一个命名 getter；
3. 在 `ConfigManager.cpp` 中实现，用 `getInt/getString/getDouble/getBool` 返回，并带上默认值；
4. 在业务代码中调用该 getter 取值。

> `getDouble` 对整数 JSON 值同样兼容；`getInt` 仅接受整数字面量（`3`），不接受 `3.0`。
