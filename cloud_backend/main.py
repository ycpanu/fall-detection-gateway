import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
import paho.mqtt.client as mqtt
from pydantic import BaseModel

# 1. 全局配置与日志初始化
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CloudBackend")

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "fall_detection_gateway/alerts"

# 模拟内存数据库，用于存储历史报警记录（实际项目应存入 MySQL / InfluxDB）
alert_database = []
alert_lock = threading.Lock()

# 2. Pydantic 数据模型定义
class FallData(BaseModel):
    trigger_x: int
    trigger_y: int
    status: str

class AlertEvent(BaseModel):
    device_id: str
    event_type: str
    timestamp: int
    data: FallData

# 3. MQTT 客户端回调逻辑
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        logger.error("连接 MQTT Broker 失败，返回码: %s", reason_code)
    else:
        logger.info("成功连接到 MQTT 云端 Broker: %s", MQTT_BROKER)
        client.subscribe(MQTT_TOPIC)
        logger.info("已订阅报警频道: %s", MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        # 解析网关发来的 JSON 数据包
        payload_str = msg.payload.decode("utf-8")
        payload_json = json.loads(payload_str)

        # 用 Pydantic 模型校验数据结构
        alert_event = AlertEvent(**payload_json)
        logger.info("接收到边缘网关报警数据: %s", payload_json)

        # 存入数据库 (此处以存入内存列表模拟)
        with alert_lock:
            alert_database.append(alert_event)

        # 触发预警下发逻辑
        trigger_notification_service(alert_event)

    except Exception as e:
        logger.error("解析 MQTT 消息时发生错误: %s", e)

def trigger_notification_service(alert_event: AlertEvent):
    """
    模拟调用外部通知服务。
    实际开发中，应在此处对接微信小程序推送或阿里云/腾讯云 SMS 短信 API。
    """
    logger.info("正在向家属微信小程序发送设备 [%s] 的摔倒预警推送...", alert_event.device_id)

# 4. FastAPI 生命周期管理 (整合 MQTT)
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：连接 MQTT Broker 并开启独立后台网络循环
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()  # 在后台线程中非阻塞运行 MQTT
    except Exception as e:
        logger.error("MQTT 初始化失败: %s", e)

    yield  # 交出控制权，FastAPI 主程序运行

    # 关闭时：安全断开 MQTT 连接
    logger.info("正在关闭云端后端服务，断开 MQTT 连接...")
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except Exception as e:
        logger.error("断开 MQTT 连接时发生错误: %s", e)

app = FastAPI(title="Fall Detection Cloud Backend", lifespan=lifespan)

# 5. RESTful API 接口 (供 Vue/React 后台或小程序调用)
@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "云端接收服务正在运行"}

@app.get("/api/alerts", response_model=List[AlertEvent], tags=["Alerts"])
async def get_history_alerts():
    """
    提供给后台管理端或家属小程序拉取历史摔倒报警记录的接口
    """
    with alert_lock:
        return list(alert_database)


'''
启动步骤

1. 安装依赖（fastapi 会自动带上 pydantic；main.py 用了 CallbackAPIVersion.VERSION2，需要 paho-mqtt ≥ 2.0）

pip install fastapi "uvicorn[standard]" paho-mqtt

2. 启动服务（在 apps/ 目录下执行）

cd apps
uvicorn main:app --host 0.0.0.0 --port 8000

开发时想改代码自动重载，加 --reload：

uvicorn main:app --reload

3. 验证

┌──────────────────────────────────┬────────────────────────────────────┐
│               地址               │                说明                │
├──────────────────────────────────┼────────────────────────────────────┤
│ http://localhost:8000/           │ 健康检查，返回 {"status":"ok",...} │
├──────────────────────────────────┼────────────────────────────────────┤
│ http://localhost:8000/api/alerts │ 拉取历史报警记录                   │
├──────────────────────────────────┼────────────────────────────────────┤
│ http://localhost:8000/docs       │ Swagger 接口文档（自动生成）       │
└──────────────────────────────────┴────────────────────────────────────┘

启动后日志会显示「成功连接到 MQTT 云端 Broker」并订阅报警频道，此时网关发的摔倒报警就能被接收、存入内存列表。
'''
